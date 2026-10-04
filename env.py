"""
Vectorized "skill chain" environment from the paper's toy experiments (Sec. 4 / App. B.2).

An episode lasts H steps and is split into B = H / L "blocks" (task stages).
State  : the index of the block the agent is currently in.
Action : one of A discrete actions. Each block b has a secret target action a*_b.
Reward : +0.5 if the action is correct for the current block, -0.5 otherwise.
Gating : after L steps in a block, the agent moves to the next block if it made
         >= k_mastery correct actions there, OR with probability p_prog anyway.
Episode ends when H steps have elapsed (deterministic horizon -> synchronised resets).

All N environments live in a few torch tensors, so ONE call to step() advances
all of them at once on the GPU (or CPU). No Python loop over environments.
"""
import random
import torch


class ChainEnv:
    def __init__(self, num_envs, horizon=200, block_len=5, num_actions=20,
                 p_prog=0.5, k_mastery=3, task_seed=0, device="cpu"):
        self.N, self.H, self.L, self.A = num_envs, horizon, block_len, num_actions
        self.B = horizon // block_len
        self.p_prog, self.k_mastery, self.device = p_prog, k_mastery, device
        g = torch.Generator().manual_seed(task_seed)          # the task itself
        self.targets = torch.randint(0, num_actions, (self.B,), generator=g).to(device)
        z = lambda: torch.zeros(num_envs, dtype=torch.long, device=device)
        self.block, self.t, self.steps_in_block, self.correct_in_block = z(), z(), z(), z()

    def reset(self):
        self._reset_where(torch.ones(self.N, dtype=torch.bool, device=self.device))
        return self.block.clone()

    def _reset_where(self, mask):
        for x in (self.block, self.t, self.steps_in_block, self.correct_in_block):
            x[mask] = 0

    def step(self, actions, active=None):
        """Advance every env by one step. `active` (bool mask) lets us step only a
        subset of envs -- used for the staggered warm-up."""
        if active is None:
            active = torch.ones(self.N, dtype=torch.bool, device=self.device)
        a = active.long()
        correct = actions == self.targets[self.block]
        reward = torch.where(correct, 0.5, -0.5) * active
        self.t += a
        self.steps_in_block += a
        self.correct_in_block += (correct & active).long()

        block_over = active & (self.steps_in_block >= self.L)
        mastered = self.correct_in_block >= self.k_mastery
        lucky = torch.rand(self.N, device=self.device) < self.p_prog
        advance = block_over & (mastered | lucky)
        self.block = torch.where(advance, (self.block + 1).clamp(max=self.B - 1), self.block)
        self.steps_in_block[block_over] = 0
        self.correct_in_block[block_over] = 0

        done = active & (self.t >= self.H)
        success = done & (self.block == self.B - 1)
        final_block = torch.where(done, self.block, torch.zeros_like(self.block))
        self._reset_where(done)                     # auto-reset finished envs
        return self.block.clone(), reward, done, {"success": success, "final_block": final_block}


def stagger(env, rollout_len):
    """Staggered resets (paper Sec. 3.3 / App. B.1).
    Split envs into N_B = ceil(H/K) groups; group j is advanced j*K steps with
    random actions before training starts. Afterwards all envs are stepped
    synchronously as usual, but they sit at different points of the horizon."""
    n_groups = -(-env.H // rollout_len)
    offsets = (torch.arange(env.N, device=env.device) % n_groups) * rollout_len
    for w in range(int(offsets.max())):
        rand_a = torch.randint(0, env.A, (env.N,), device=env.device)
        env.step(rand_a, active=offsets > w)
    return offsets


class PythonChainEnv:
    """Same env, one Python object per environment. Only used as a slow
    baseline in the throughput benchmark ("classic CPU-style" vectorization)."""
    def __init__(self, horizon=200, block_len=5, num_actions=20, p_prog=0.5, k_mastery=3, targets=None):
        self.H, self.L, self.A, self.p, self.k = horizon, block_len, num_actions, p_prog, k_mastery
        self.B = horizon // block_len
        self.targets = targets or [random.randrange(num_actions) for _ in range(self.B)]
        self.reset()

    def reset(self):
        self.block = self.t = self.sib = self.cib = 0
        return self.block

    def step(self, a):
        correct = a == self.targets[self.block]
        self.t += 1; self.sib += 1; self.cib += correct
        if self.sib >= self.L:
            if self.cib >= self.k or random.random() < self.p:
                self.block = min(self.block + 1, self.B - 1)
            self.sib = self.cib = 0
        done = self.t >= self.H
        if done:
            self.reset()
        return self.block, (0.5 if correct else -0.5), done
