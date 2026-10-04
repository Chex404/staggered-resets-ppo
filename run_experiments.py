"""
Runs all experiments and saves raw logs to results/*.json.
    python run_experiments.py --preset quick     (~3-5 min on CPU, <2 min on GPU)
    python run_experiments.py                    (default, ~30 min on 1 CPU core, much less on GPU)

Experiments
  E1 main      : sync vs staggered, H=200, K=5, N=512 (paper's toy setting), several seeds
  E2 horizon   : vary horizon H with K fixed  -> effect grows as K << H   (paper Fig. 2a)
  E3 scaling   : vary number of parallel envs N  (paper Fig. 5, scalability)
  E4 benchmark : raw env / rollout throughput, python loop vs vectorized CPU vs GPU
"""
import argparse, json, os, time
from dataclasses import replace
from ppo import Config, train
import benchmark

PRESETS = {
    "quick":   dict(seeds=[0, 1], updates=100, horizons=[50, 200], sweep_seeds=[0], envs=[128, 512, 1024]),
    "default": dict(seeds=[0, 1, 2], updates=150, horizons=[50, 100, 200, 400], sweep_seeds=[0, 1], envs=[128, 512, 2048]),
}


def save(obj, name):
    os.makedirs("results", exist_ok=True)
    with open(f"results/{name}.json", "w") as f:
        json.dump(obj, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="default", choices=PRESETS)
    ap.add_argument("--only", default="main,horizon,scaling,benchmark")
    args = ap.parse_args()
    P, only = PRESETS[args.preset], args.only.split(",")
    base = Config(num_updates=P["updates"])
    print(f"device = {base.device}, preset = {args.preset}")
    t0 = time.time()

    if "main" in only:
        runs = [train(replace(base, reset_mode=m, seed=s)) for s in P["seeds"] for m in ("sync", "staggered")]
        save(runs, "main")

    if "horizon" in only:
        runs = []
        for H in P["horizons"]:
            for s in P["sweep_seeds"]:
                for m in ("sync", "staggered"):
                    runs.append(train(replace(base, reset_mode=m, seed=s, horizon=H), verbose=False))
                    r = runs[-1]["eval"][-1]
                    print(f"  H={H:4d} {m:9s} seed={s}: stage_acc={sum(r['block_acc'])/len(r['block_acc']):.2f}")
        save(runs, "horizon")

    if "scaling" in only:
        runs = []
        for N in P["envs"]:
            for m in ("sync", "staggered"):
                runs.append(train(replace(base, reset_mode=m, num_envs=N, eval_every=2), verbose=False))
                print(f"  N={N:5d} {m:9s}: {runs[-1]['total_time']:.1f}s, {runs[-1]['sps']:.0f} steps/s")
        save(runs, "scaling")

    if "benchmark" in only:
        save(benchmark.run(quick=args.preset == "quick"), "benchmark")

    print(f"All done in {time.time() - t0:.0f}s. Now run: python make_plots.py")


if __name__ == "__main__":
    main()
