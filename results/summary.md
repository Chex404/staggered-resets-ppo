# Results summary

## E1. Main comparison (N=512, K=5, H=200, 3 seeds, 150 updates, device=cpu)

| Metric | Synchronous | Staggered |
|---|---|---|
| Final eval return (max = +H/2) | 39.355 ± 3.904 | 73.472 ± 10.075 |
| Final accuracy over all task stages | 0.703 ± 0.024 | 0.933 ± 0.024 |
| Final success rate | 0.005 ± 0.001 | 0.372 ± 0.149 |
| Batch-accuracy oscillation (std of train acc, 2nd half) | 0.295 ± 0.006 | 0.046 ± 0.016 |
| Mean critic error (value MSE) | 2.524 ± 0.065 | 1.861 ± 0.050 |
| Max critic error (value MSE) | 112.421 ± 6.600 | 3.729 ± 0.074 |
| Mean approx KL per update | 0.079 ± 0.025 | 0.030 ± 0.012 |
| Mean forgetting | 0.126 ± 0.022 | 0.016 ± 0.010 |
| Training throughput (steps/s) | 2004.737 ± 340.269 | 1162.714 ± 818.717 |
| Total training time (s) | 196.608 ± 29.822 | 23284.629 ± 32617.452 |

## E2. Horizon sweep (K=5 fixed, H/K = number of distinct batch 'slices')

| H | H/K | stage acc (sync) | stage acc (staggered) | return (sync) | return (staggered) |
|---|---|---|---|---|---|
| 50 | 10 | 0.950 ± 0.050 | 1.000 ± 0.000 | 21.230 ± 3.770 | 25.000 ± 0.000 |
| 100 | 20 | 0.900 ± 0.000 | 0.975 ± 0.025 | 36.416 ± 0.212 | 45.133 ± 4.867 |
| 200 | 40 | 0.688 ± 0.012 | 0.925 ± 0.025 | 36.803 ± 1.823 | 69.492 ± 10.234 |
| 400 | 80 | 0.433 ± 0.046 | 0.863 ± 0.000 | -32.243 ± 15.283 | 138.916 ± 0.492 |

## E3. Scaling with number of parallel environments N (threshold: stage accuracy ≥ 0.8)

| N | mode | steps/s | final stage acc | env steps to threshold | wall s to threshold |
|---|---|---|---|---|---|
| 128 | sync | 469 | 0.650 | not reached | not reached |
| 512 | sync | 1584 | 0.667 | not reached | not reached |
| 2048 | sync | 4204 | 0.658 | not reached | not reached |
| 128 | staggered | 477 | 0.800 | 79360 | 164.8 |
| 512 | staggered | 1622 | 0.975 | 174080 | 102.5 |
| 2048 | staggered | 3454 | 0.950 | 491520 | 142.4 |

(`not reached` = did not reach the threshold within the training budget.)

## E4. Throughput benchmark (env steps / second)

| N | Python loop | torch cpu env | torch cpu env+policy |
|---|---|---|---|
| 16 | 321,040 | 10,923 | 2,368 |
| 256 | 361,077 | 265,352 | 18,184 |
| 1024 | 375,665 | 862,714 | 46,170 |
| 4096 | 392,473 | 2,625,897 | 85,152 |
| 16384 | — | 5,106,041 | 118,037 |
| 65536 | — | 5,544,059 | 161,799 |
| 262144 | — | 12,269,383 | 180,679 |

Reset load (N=512, H=200, K=5): synchronous = max 512 resets in a single step (spiky), staggered = max 13 per step (smooth). Total resets are equal: 1536 vs 1536.
