"""
Accelerated-AI benchmark: how many environment steps per second can we simulate?
  (a) Python loop   : one Python object per env, stepped one after another
  (b) Torch CPU     : all envs as tensors, one vectorized step on CPU
  (c) Torch GPU     : same code on CUDA (if available)
Also measures "rollout" throughput = env step + policy forward pass (what PPO really does),
and the reset load per step for sync vs staggered resets.
"""
import time, random
import torch
from env import ChainEnv, PythonChainEnv, stagger
from ppo import Agent


def sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()


def bench_python(n_envs, steps):
    envs = [PythonChainEnv() for _ in range(n_envs)]
    t = time.perf_counter()
    for _ in range(steps):
        for e in envs:
            e.step(random.randrange(20))
    return n_envs * steps / (time.perf_counter() - t)


def bench_torch(n_envs, steps, dev, with_policy=False):
    env = ChainEnv(n_envs, device=dev)
    obs = env.reset()
    agent = Agent(env.B, 20, 256, 4).to(dev) if with_policy else None
    with torch.no_grad():
        for i in range(steps + 5):          # 5 warm-up steps (CUDA init, allocator)
            if i == 5:
                sync(dev); t = time.perf_counter()
            a = agent.act(obs)[0] if with_policy else torch.randint(0, 20, (n_envs,), device=dev)
            obs, *_ = env.step(a)
    sync(dev)
    return n_envs * steps / (time.perf_counter() - t)


def reset_profile(n_envs=512, K=5, H=200, steps=600):
    """Number of env resets happening at each global step."""
    out = {}
    for mode in ("sync", "staggered"):
        env = ChainEnv(n_envs, horizon=H, p_prog=1.0)
        env.reset()
        if mode == "staggered":
            stagger(env, K)
        counts = []
        for _ in range(steps):
            _, _, done, _ = env.step(torch.randint(0, 20, (n_envs,)))
            counts.append(int(done.sum()))
        out[mode] = counts
    return out


def run(quick=False):
    devs = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    sizes = [16, 256, 4096, 65536] if quick else [16, 256, 1024, 4096, 16384, 65536, 262144]
    res = {"sizes": sizes, "devices": devs, "python": {}, "env": {d: {} for d in devs},
           "rollout": {d: {} for d in devs}}
    if torch.cuda.is_available():
        res["gpu_name"] = torch.cuda.get_device_name(0)
    for n in sizes:
        if n <= 4096:   # python loop gets too slow beyond this
            res["python"][n] = bench_python(n, max(20, 200000 // n))
        for d in devs:
            res["env"][d][n] = bench_torch(n, 50, d)
            res["rollout"][d][n] = bench_torch(n, 20, d, with_policy=True)
        print(f"  N={n:7d} | python {res['python'].get(n, float('nan')):12.0f} | "
              + " | ".join(f"{d} env {res['env'][d][n]:12.0f} rollout {res['rollout'][d][n]:11.0f}" for d in devs)
              + "  steps/s")
    res["reset_profile"] = reset_profile()
    return res


if __name__ == "__main__":
    run(quick=True)
