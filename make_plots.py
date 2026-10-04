"""
Reads results/*.json and writes figures to figures/ and a table summary to results/summary.md.
    python make_plots.py
Figures are produced for whichever experiments have results available.
"""
import json, os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from env import ChainEnv, stagger

C = {"sync": "#1f77b4", "staggered": "#d62728"}
LBL = {"sync": "Synchronous resets", "staggered": "Staggered resets"}
os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
summary = ["# Results summary\n"]


def load(name):
    p = f"results/{name}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def stage_acc(e):
    return float(np.mean(e["block_acc"]))


def final(run, key, n=3):
    """Average of the last n evaluations (more robust than a single point)."""
    ev = run["eval"][-n:]
    return float(np.mean([stage_acc(e) if key == "stage_acc" else e[key] for e in ev]))


def by_mode(runs):
    return {m: [r for r in runs if r["config"]["reset_mode"] == m] for m in ("sync", "staggered")}


def ms(x):
    return f"{np.mean(x):.3f} ± {np.std(x):.3f}"


# --------------------------------------------------------------------------------------
# Fig 1: what staggering does to the environments (no training needed, pure simulation)
# --------------------------------------------------------------------------------------
def fig_reset_schedule(N=64, H=200, K=5, steps=500):
    fig, ax = plt.subplots(2, 2, figsize=(13, 6.8), gridspec_kw={"height_ratios": [3, 1.2]})
    for col, mode in enumerate(("sync", "staggered")):
        torch.manual_seed(0)
        env = ChainEnv(N, horizon=H, p_prog=1.0)
        env.reset()
        if mode == "staggered":
            stagger(env, K)
        T, resets = np.zeros((N, steps)), []
        for s in range(steps):
            T[:, s] = env.t.numpy()
            _, _, done, _ = env.step(torch.randint(0, 20, (N,)))
            resets.append(int(done.sum()))
        im = ax[0, col].imshow(T, aspect="auto", cmap="viridis", interpolation="nearest")
        for b in range(0, steps, K * 8):
            ax[0, col].axvspan(b, b + K, color="white", alpha=0.35, lw=0)
        ax[0, col].set_title(f"{LBL[mode]}\nepisode timestep per env (white = a PPO batch of K={K} steps)", fontsize=9.5)
        ax[0, col].set_ylabel("environment index")
        ax[0, col].grid(False)
        ax[1, col].bar(range(steps), resets, color=C[mode], width=1.0)
        ax[1, col].set_ylabel("# resets this step")
        ax[1, col].set_xlabel("global simulation step")
        ax[1, col].set_ylim(0, N * 1.05)
    fig.colorbar(im, ax=ax[0, :], label="timestep within episode (0 → H)")
    fig.suptitle(f"How the two reset strategies differ  (N={N} envs shown, horizon H={H}, rollout K={K})", fontweight="bold")
    fig.savefig("figures/fig1_reset_schedule.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Main experiment figures
# --------------------------------------------------------------------------------------
def fig_main(runs):
    M = by_mode(runs)
    cfg = runs[0]["config"]
    K, H = cfg["rollout_len"], cfg["horizon"]

    # Fig 2: batch composition over training (paper Fig. 6c/d)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for i, m in enumerate(("sync", "staggered")):
        h = np.array(M[m][0]["batch_t_hist"]).T
        im = ax[i].imshow(np.maximum(h, 1e-3), aspect="auto", origin="lower", cmap="magma", interpolation="nearest",
                          extent=[1, len(h[0]), 0, H], norm=LogNorm(vmin=1e-3, vmax=1))
        ax[i].set_title(f"{LBL[m]}\nepisode timesteps contained in each PPO batch", fontsize=10)
        ax[i].set_xlabel("PPO update")
        ax[i].grid(False)
    ax[0].set_ylabel("episode timestep of batch samples")
    fig.colorbar(im, ax=ax, label="fraction of batch (log scale)")
    fig.savefig("figures/fig2_batch_composition.png", bbox_inches="tight")
    plt.close(fig)

    # Fig 3: learning curves
    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    for m, rs in M.items():
        x = np.array([e["step"] for e in rs[0]["eval"]])
        for j, (key, name) in enumerate([("return", "Eval episode return"), ("stage_acc", "Policy accuracy over ALL task stages"),
                                         ("success", "Eval success rate (reach final stage)")]):
            y = np.array([[stage_acc(e) if key == "stage_acc" else e[key] for e in r["eval"]] for r in rs])
            ax[j].plot(x, y.mean(0), color=C[m], label=LBL[m])
            ax[j].fill_between(x, y.mean(0) - y.std(0), y.mean(0) + y.std(0), color=C[m], alpha=0.2)
            ax[j].set_title(name); ax[j].set_xlabel("environment steps")
            ax[j].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax[3].plot(np.arange(1, len(rs[0]["train_acc"]) + 1), rs[0]["train_acc"], color=C[m], label=LBL[m], lw=1)
    ax[3].set_title("Accuracy on the current training batch (seed 0)")
    ax[3].set_xlabel("PPO update")
    for u in range(H // K, len(M["sync"][0]["train_acc"]), H // K):
        ax[3].axvline(u, color="gray", ls=":", lw=1)
    ax[0].legend()
    fig.suptitle(f"PPO learning: mean ± std over {len(M['sync'])} seeds  (N={cfg['num_envs']}, K={K}, H={H})", fontweight="bold")
    fig.tight_layout()
    fig.savefig("figures/fig3_learning_curves.png", bbox_inches="tight")
    plt.close(fig)

    # Fig 4: training stability (paper App. E)
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    for m, rs in M.items():
        for j, key in enumerate(("value_mse", "approx_kl")):
            y = np.array([r[key] for r in rs]).mean(0)
            ax[j].plot(np.arange(1, len(y) + 1), y, color=C[m], label=LBL[m], lw=1.2)
    for a in ax:
        for u in range(H // K, len(M["sync"][0]["value_mse"]), H // K):
            a.axvline(u, color="gray", ls=":", lw=1)
        a.set_xlabel("PPO update")
    ax[0].set_title("Critic error on each new batch (value MSE)\n(dotted = synchronous mass reset every H/K updates)")
    ax[1].set_title("Policy change per update (approx. KL)")
    ax[0].set_yscale("log"); ax[1].set_yscale("log")
    ax[0].legend()
    fig.tight_layout()
    fig.savefig("figures/fig4_stability.png", bbox_inches="tight")
    plt.close(fig)

    # Fig 5: forgetting matrix (paper App. E)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for i, m in enumerate(("sync", "staggered")):
        As = np.array([[e["block_acc"] for e in r["eval"]] for r in M[m]])          # (seeds, evals, blocks)
        F = (np.maximum.accumulate(As, axis=1) - As).mean(0)                       # forgetting, averaged over seeds
        upd = [e["update"] for e in M[m][0]["eval"]]
        im = ax[i].imshow(F.T, aspect="auto", origin="lower", cmap="inferno", vmin=0, vmax=1,
                          extent=[upd[0], upd[-1], 0, F.shape[1]], interpolation="nearest")
        ax[i].set_title(f"{LBL[m]}\nforgetting = best-so-far − current acc. (mean {F.mean():.3f})", fontsize=10)
        ax[i].set_xlabel("PPO update"); ax[i].grid(False)
    ax[0].set_ylabel("task stage (block index)")
    fig.colorbar(im, ax=ax, label="accuracy lost")
    fig.savefig("figures/fig5_forgetting.png", bbox_inches="tight")
    plt.close(fig)

    # Fig 6: compute — return vs wall-clock + training throughput
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    for m, rs in M.items():
        for r in rs:
            w = [r["wall"][e["update"] - 1] for e in r["eval"]]
            ax[0].plot(w, [e["return"] for e in r["eval"]], color=C[m], alpha=0.7,
                       label=LBL[m] if r is rs[0] else None)
    ax[0].set_xlabel(f"wall-clock seconds ({cfg['device']})"); ax[0].set_title("Eval return vs wall-clock time"); ax[0].legend()
    sps = [[r["sps"] for r in M[m]] for m in ("sync", "staggered")]
    ax[1].bar([LBL["sync"], LBL["staggered"]], [np.mean(s) for s in sps], yerr=[np.std(s) for s in sps],
              color=[C["sync"], C["staggered"]])
    ax[1].set_title("Training throughput (env steps / s, incl. PPO updates)")
    fig.tight_layout()
    fig.savefig("figures/fig6_walltime_throughput.png", bbox_inches="tight")
    plt.close(fig)

    # Summary table
    summary.append(f"## E1. Main comparison (N={cfg['num_envs']}, K={K}, H={H}, {len(M['sync'])} seeds, "
                   f"{cfg['num_updates']} updates, device={cfg['device']})\n")
    summary.append("| Metric | Synchronous | Staggered |\n|---|---|---|")
    rows = [
        ("Final eval return (max = +H/2)", lambda r: final(r, "return")),
        ("Final accuracy over all task stages", lambda r: final(r, "stage_acc")),
        ("Final success rate", lambda r: final(r, "success")),
        ("Batch-accuracy oscillation (std of train acc, 2nd half)", lambda r: np.std(r["train_acc"][len(r["train_acc"]) // 2:])),
        ("Mean critic error (value MSE)", lambda r: np.mean(r["value_mse"])),
        ("Max critic error (value MSE)", lambda r: np.max(r["value_mse"])),
        ("Mean approx KL per update", lambda r: np.mean(r["approx_kl"])),
        ("Mean forgetting", lambda r: float((np.maximum.accumulate(np.array([e["block_acc"] for e in r["eval"]]), 0)
                                             - np.array([e["block_acc"] for e in r["eval"]])).mean())),
        ("Training throughput (steps/s)", lambda r: r["sps"]),
        ("Total training time (s)", lambda r: r["total_time"]),
    ]
    for name, f in rows:
        summary.append(f"| {name} | {ms([f(r) for r in M['sync']])} | {ms([f(r) for r in M['staggered']])} |")
    summary.append("")


def fig_horizon(runs):
    M = by_mode(runs)
    Hs = sorted({r["config"]["horizon"] for r in runs})
    K = runs[0]["config"]["rollout_len"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    summary.append(f"## E2. Horizon sweep (K={K} fixed, H/K = number of distinct batch 'slices')\n")
    summary.append("| H | H/K | stage acc (sync) | stage acc (staggered) | return (sync) | return (staggered) |\n|---|---|---|---|---|---|")
    stats = {}
    for m in M:
        for j, key in enumerate(("stage_acc", "return")):
            vals = [[final(r, key) for r in M[m] if r["config"]["horizon"] == H] for H in Hs]
            stats[(m, key)] = vals
            ax[j].errorbar(Hs, [np.mean(v) for v in vals], yerr=[np.std(v) for v in vals], color=C[m],
                           marker="o", capsize=3, label=LBL[m])
    for i, H in enumerate(Hs):
        summary.append(f"| {H} | {H // K} | {ms(stats[('sync','stage_acc')][i])} | {ms(stats[('staggered','stage_acc')][i])} "
                       f"| {ms(stats[('sync','return')][i])} | {ms(stats[('staggered','return')][i])} |")
    summary.append("")
    ax[0].set_title("Final accuracy over all task stages"); ax[1].set_title("Final eval return (normalised scale differs per H)")
    for a in ax:
        a.set_xscale("log"); a.set_xticks(Hs); a.set_xticklabels(Hs); a.set_xlabel(f"episode horizon H  (rollout K={K})")
    ax[0].legend()
    fig.suptitle("The longer the horizon relative to the rollout (K ≪ H), the more synchronous resets hurt", fontweight="bold")
    fig.tight_layout()
    fig.savefig("figures/fig7_horizon_sweep.png", bbox_inches="tight")
    plt.close(fig)


def fig_scaling(runs, thr=0.8):
    M = by_mode(runs)
    Ns = sorted({r["config"]["num_envs"] for r in runs})
    fig, ax = plt.subplots(1, 3, figsize=(16, 3.8))
    summary.append(f"## E3. Scaling with number of parallel environments N (threshold: stage accuracy ≥ {thr})\n")
    summary.append("| N | mode | steps/s | final stage acc | env steps to threshold | wall s to threshold |\n|---|---|---|---|---|---|")
    for m in M:
        steps_to, wall_to, fin = [], [], []
        for N in Ns:
            r = [r for r in M[m] if r["config"]["num_envs"] == N][0]
            hit = [e for e in r["eval"] if stage_acc(e) >= thr]
            steps_to.append(hit[0]["step"] if hit else np.nan)
            wall_to.append(r["wall"][hit[0]["update"] - 1] if hit else np.nan)
            fin.append(final(r, "stage_acc"))
            summary.append(f"| {N} | {m} | {r['sps']:.0f} | {fin[-1]:.3f} | "
                           f"{'%d' % steps_to[-1] if hit else 'not reached'} | {'%.1f' % wall_to[-1] if hit else 'not reached'} |")
        ax[0].plot(Ns, fin, "o-", color=C[m], label=LBL[m])
        ax[1].plot(Ns, steps_to, "o-", color=C[m], label=LBL[m])
        ax[2].plot(Ns, wall_to, "o-", color=C[m], label=LBL[m])
    summary.append("\n(`not reached` = did not reach the threshold within the training budget.)\n")
    ax[0].set_title(f"Final stage accuracy (same # of PPO updates)")
    ax[1].set_title(f"Env steps to reach {thr:.0%} stage accuracy")
    ax[2].set_title(f"Wall-clock s to reach {thr:.0%} stage accuracy")
    for a in ax:
        a.set_xscale("log", base=2); a.set_xlabel("parallel environments N")
    ax[0].legend()
    fig.tight_layout()
    fig.savefig("figures/fig8_scaling.png", bbox_inches="tight")
    plt.close(fig)


def fig_benchmark(b):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    py = {int(k): v for k, v in b["python"].items()}
    ax[0].plot(list(py), list(py.values()), "s--", color="gray", label="Python loop (1 object per env)")
    for d, st in zip(b["devices"], ("o-", "^-")):
        e = {int(k): v for k, v in b["env"][d].items()}
        r = {int(k): v for k, v in b["rollout"][d].items()}
        ax[0].plot(list(e), list(e.values()), st, label=f"Vectorized torch, {d}")
        ax[1].plot(list(r), list(r.values()), st, label=f"env + policy network, {d}")
    ax[0].set_title("Environment simulation throughput")
    ax[1].set_title("Rollout throughput (what PPO data collection costs)")
    for a in ax:
        a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("parallel environments N"); a.set_ylabel("env steps / second"); a.legend()
    fig.suptitle("Vectorization / GPU acceleration" + (f"  ({b['gpu_name']})" if "gpu_name" in b else "  (CPU only on this machine)"),
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig("figures/fig9_throughput.png", bbox_inches="tight")
    plt.close(fig)

    summary.append("## E4. Throughput benchmark (env steps / second)\n")
    hdr = "| N | Python loop | " + " | ".join(f"torch {d} env | torch {d} env+policy" for d in b["devices"]) + " |"
    summary.append(hdr + "\n|" + "---|" * (2 + 2 * len(b["devices"])))
    for n in b["sizes"]:
        cells = [f"{py.get(n, float('nan')):,.0f}" if n in py else "—"]
        for d in b["devices"]:
            cells += [f"{b['env'][d][str(n)]:,.0f}", f"{b['rollout'][d][str(n)]:,.0f}"]
        summary.append(f"| {n} | " + " | ".join(cells) + " |")
    rp = b["reset_profile"]
    summary.append(f"\nReset load (N=512, H=200, K=5): synchronous = max {max(rp['sync'])} resets in a single step "
                   f"(spiky), staggered = max {max(rp['staggered'])} per step (smooth). "
                   f"Total resets are equal: {sum(rp['sync'])} vs {sum(rp['staggered'])}.\n")


if __name__ == "__main__":
    fig_reset_schedule()
    if (r := load("main")): fig_main(r)
    if (r := load("horizon")): fig_horizon(r)
    if (r := load("scaling")): fig_scaling(r)
    if (r := load("benchmark")): fig_benchmark(r)
    open("results/summary.md", "w").write("\n".join(summary))
    print("\n".join(summary))
    print("\nFigures written to figures/")
