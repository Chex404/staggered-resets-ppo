"""
Minimal PPO (CleanRL style) for the vectorized ChainEnv.
The ONLY difference between the two methods is `reset_mode`:
    "sync"      -> all N envs start at t=0 together (standard practice)
    "staggered" -> envs are spread over the horizon before training (the paper)
Hyperparameters follow the paper's Table 2 (toy environment).
"""
import time
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn
from env import ChainEnv, stagger


@dataclass
class Config:
    reset_mode: str = "sync"      # "sync" | "staggered"
    num_envs: int = 512           # N  (parallel environments)
    rollout_len: int = 5          # K  (steps per env per PPO update)
    horizon: int = 200            # H  (episode length)
    block_len: int = 5            # L
    num_actions: int = 20
    p_prog: float = 0.5
    k_mastery: int = 3
    num_updates: int = 150
    epochs: int = 4
    minibatches: int = 4
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    hidden: int = 256
    layers: int = 4
    seed: int = 0
    eval_every: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def mlp(n_blocks, hidden, layers, out):
    mods = [nn.Embedding(n_blocks, 64), nn.Linear(64, hidden), nn.ReLU()]
    for _ in range(layers - 1):
        mods += [nn.Linear(hidden, hidden), nn.ReLU()]
    return nn.Sequential(*mods, nn.Linear(hidden, out))


class Agent(nn.Module):
    def __init__(self, n_blocks, n_actions, hidden, layers):
        super().__init__()
        self.actor = mlp(n_blocks, hidden, layers, n_actions)
        self.critic = mlp(n_blocks, hidden, layers, 1)

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    def act(self, obs, action=None):
        dist = torch.distributions.Categorical(logits=self.actor(obs))
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.value(obs)


@torch.no_grad()
def evaluate(agent, cfg, n_eval=256):
    """Fresh envs, greedy policy, one full episode -> return / success / progress.
    Plus exact per-block accuracy (policy is a function of the block index only)."""
    env = ChainEnv(n_eval, cfg.horizon, cfg.block_len, cfg.num_actions, cfg.p_prog,
                   cfg.k_mastery, task_seed=cfg.seed, device=cfg.device)
    obs, ret = env.reset(), torch.zeros(n_eval, device=cfg.device)
    for _ in range(cfg.horizon):
        obs, r, done, info = env.step(agent.actor(obs).argmax(-1))
        ret += r
    blocks = torch.arange(env.B, device=cfg.device)
    block_acc = (agent.actor(blocks).argmax(-1) == env.targets).float()
    return {"return": ret.mean().item(), "success": info["success"].float().mean().item(),
            "progress": (info["final_block"].float() / (env.B - 1)).mean().item(),
            "block_acc": block_acc.cpu().tolist()}


def train(cfg: Config, verbose=True):
    torch.manual_seed(cfg.seed)
    dev = cfg.device
    env = ChainEnv(cfg.num_envs, cfg.horizon, cfg.block_len, cfg.num_actions, cfg.p_prog,
                   cfg.k_mastery, task_seed=cfg.seed, device=dev)
    agent = Agent(env.B, cfg.num_actions, cfg.hidden, cfg.layers).to(dev)
    opt = torch.optim.Adam(agent.parameters(), lr=cfg.lr, eps=1e-5)

    obs = env.reset()
    if cfg.reset_mode == "staggered":
        stagger(env, cfg.rollout_len)       # warm-up steps are NOT counted as training data
        obs = env.block.clone()

    K, N = cfg.rollout_len, cfg.num_envs
    buf = {k: torch.zeros(K, N, device=dev) for k in ("logp", "rew", "done", "val")}
    buf_obs = torch.zeros(K, N, dtype=torch.long, device=dev)
    buf_act = torch.zeros(K, N, dtype=torch.long, device=dev)
    buf_t = torch.zeros(K, N, dtype=torch.long, device=dev)

    log = {k: [] for k in ("step", "wall", "train_acc", "value_mse", "approx_kl", "entropy",
                           "batch_block_hist", "batch_t_hist", "ep_return", "ep_success")}
    log["eval"] = []
    ep_ret = torch.zeros(N, device=dev)
    finished_returns, finished_success = [], []
    t_bins = -(-cfg.horizon // K)
    global_step, t0 = 0, time.perf_counter()

    for update in range(1, cfg.num_updates + 1):
        # ---------------- 1. rollout: N envs x K steps, all in parallel ----------------
        with torch.no_grad():
            for k in range(K):
                buf_t[k] = env.t
                a, logp, _, v = agent.act(obs)
                buf_obs[k], buf_act[k], buf["logp"][k], buf["val"][k] = obs, a, logp, v
                obs, r, done, info = env.step(a)
                buf["rew"][k], buf["done"][k] = r, done.float()
                ep_ret += r
                if done.any():
                    finished_returns += ep_ret[done].tolist()
                    finished_success += info["success"][done].float().tolist()
                    ep_ret[done] = 0
            global_step += K * N

            # ---------------- 2. GAE advantages ----------------
            next_v = agent.value(obs)
            adv = torch.zeros(K, N, device=dev)
            last = torch.zeros(N, device=dev)
            for k in reversed(range(K)):
                nv = next_v if k == K - 1 else buf["val"][k + 1]
                nonterm = 1.0 - buf["done"][k]
                delta = buf["rew"][k] + cfg.gamma * nv * nonterm - buf["val"][k]
                last = delta + cfg.gamma * cfg.gae_lambda * nonterm * last
                adv[k] = last
            ret = adv + buf["val"]
            value_mse = ((buf["val"] - ret) ** 2).mean().item()   # critic error on NEW batch

        # ---------------- 3. PPO update ----------------
        b_obs, b_act = buf_obs.flatten(), buf_act.flatten()
        b_logp, b_adv, b_ret = buf["logp"].flatten(), adv.flatten(), ret.flatten()
        mb = (K * N) // cfg.minibatches
        for _ in range(cfg.epochs):
            perm = torch.randperm(K * N, device=dev)
            for s in range(0, K * N, mb):
                i = perm[s:s + mb]
                _, new_logp, ent, v = agent.act(b_obs[i], b_act[i])
                ratio = (new_logp - b_logp[i]).exp()
                a_n = (b_adv[i] - b_adv[i].mean()) / (b_adv[i].std() + 1e-8)
                pg = torch.max(-a_n * ratio, -a_n * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)).mean()
                loss = pg + cfg.vf_coef * ((v - b_ret[i]) ** 2).mean() - cfg.ent_coef * ent.mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                opt.step()

        # ---------------- 4. logging ----------------
        with torch.no_grad():
            _, new_logp, ent, _ = agent.act(b_obs, b_act)
            kl = 0.5 * ((new_logp - b_logp) ** 2).mean().item()   # paper's policy-oscillation metric
            train_acc = (b_act == env.targets[b_obs]).float().mean().item()
        if dev == "cuda":
            torch.cuda.synchronize()
        log["step"].append(global_step)
        log["wall"].append(time.perf_counter() - t0)
        log["train_acc"].append(train_acc)
        log["value_mse"].append(value_mse)
        log["approx_kl"].append(kl)
        log["entropy"].append(ent.mean().item())
        log["batch_block_hist"].append(torch.bincount(b_obs, minlength=env.B).float().div(K * N).cpu().tolist())
        log["batch_t_hist"].append(torch.bincount(buf_t.flatten() // K, minlength=t_bins).float().div(K * N).cpu().tolist())
        log["ep_return"].append(sum(finished_returns) / len(finished_returns) if finished_returns else float("nan"))
        log["ep_success"].append(sum(finished_success) / len(finished_success) if finished_success else float("nan"))
        finished_returns, finished_success = [], []

        if update % cfg.eval_every == 0 or update == cfg.num_updates:
            e = evaluate(agent, cfg)
            e["update"], e["step"] = update, global_step
            log["eval"].append(e)
            if verbose:
                print(f"[{cfg.reset_mode:9s} seed={cfg.seed}] upd {update:4d} | steps {global_step:8d} | "
                      f"train_acc {train_acc:.2f} | eval_ret {e['return']:7.1f} | "
                      f"stage_acc {sum(e['block_acc'])/len(e['block_acc']):.2f} | "
                      f"success {e['success']:.2f} | {log['wall'][-1]:.1f}s")

    log["config"] = asdict(cfg)
    log["total_time"] = time.perf_counter() - t0
    log["sps"] = global_step / log["total_time"]
    return log


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--reset_mode", default="sync", choices=["sync", "staggered"])
    p.add_argument("--num_updates", type=int, default=150)
    p.add_argument("--num_envs", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    out = train(Config(reset_mode=a.reset_mode, num_updates=a.num_updates, num_envs=a.num_envs, seed=a.seed))
    print(f"done: {out['total_time']:.1f}s, {out['sps']:.0f} env-steps/s")
