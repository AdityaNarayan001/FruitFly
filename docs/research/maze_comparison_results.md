# Maze comparison: observed results, 21 September 2026

The current Exp 2 fly-assisted Q controller is exactly equivalent to its direct-input Q control in these trials. The new trainable reduced fly circuit learns, but the experiment does not demonstrate an anatomical advantage. The replay Q table transfers better to the tested unseen mazes.

Part A uses 400 online training episodes. Part B uses the same 20,000 randomly collected transitions and 2,000 replay updates for every controller. Do not compare percentages across the two parts as if they used the same training procedure. Each final agent/split has 2,000 evaluation trials across ten seeds and both reward assignments. Statistical intervals use ten seed units.


Engineering comparison; new plastic controller is reduced rate model, not full fly brain

These are results under a fixed experience/update budget. They do not rank every possible ML/RL model.

## Legacy comparison

| Agent | Split | Updates | Success | Mean actions | Training seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| fly_probe_q | familiar | 400 episodes | 100.0% (2000/2000) | 11.00 | — |
| q_table | familiar | 400 episodes | 100.0% (2000/2000) | 11.00 | — |
| random | familiar | 400 episodes | 31.3% (626/2000) | 85.46 | — |
| fly_probe_q | unseen | 400 episodes | 28.5% (570/2000) | 89.11 | — |
| q_table | unseen | 400 episodes | 28.5% (570/2000) | 89.11 | — |
| random | unseen | 400 episodes | 29.0% (581/2000) | 85.87 | — |

## Circuit comparison

| Agent | Split | Updates | Success | Mean actions | Training seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| fly_fixed_readout | familiar | 2000 | 37.5% (750/2000) | 73.60 | 0.59 |
| fly_plastic | familiar | 2000 | 87.5% (1750/2000) | 24.50 | 0.70 |
| q_table | familiar | 2000 | 85.0% (1700/2000) | 27.35 | 0.21 |
| random | familiar | 2000 | 31.3% (626/2000) | 85.46 | 0.00 |
| rewired_plastic | familiar | 2000 | 95.0% (1900/2000) | 16.40 | 0.71 |
| fly_fixed_readout | unseen | 2000 | 19.6% (392/2000) | 90.56 | 0.59 |
| fly_plastic | unseen | 2000 | 24.9% (498/2000) | 92.48 | 0.70 |
| q_table | unseen | 2000 | 52.3% (1046/2000) | 72.08 | 0.21 |
| random | unseen | 2000 | 29.0% (581/2000) | 85.87 | 0.00 |
| rewired_plastic | unseen | 2000 | 26.9% (538/2000) | 90.35 | 0.71 |

## Paired success-rate differences

Intervals resample training seeds, averaging the two reward assignments inside each seed. Units are percentage points.

- legacy, familiar: fly_probe_q minus q_table: 0.00 pp; 95% interval [0.00, 0.00].
- legacy, unseen: fly_probe_q minus q_table: 0.00 pp; 95% interval [0.00, 0.00].
- circuit, familiar: fly_plastic minus q_table: 2.50 pp; 95% interval [-12.50, 17.50].
- circuit, familiar: fly_plastic minus rewired_plastic: -7.50 pp; 95% interval [-15.00, 0.00].
- circuit, familiar: fly_plastic minus fly_fixed_readout: 50.00 pp; 95% interval [22.50, 75.00].
- circuit, unseen: fly_plastic minus q_table: -27.40 pp; 95% interval [-35.70, -18.05].
- circuit, unseen: fly_plastic minus rewired_plastic: -2.00 pp; 95% interval [-7.50, 3.60].
- circuit, unseen: fly_plastic minus fly_fixed_readout: 5.30 pp; 95% interval [-3.20, 14.60].

The plastic network trains modeled KC-to-MBON strengths and an external action readout. It does not reproduce natural plasticity, simulate the entire connectome, or demonstrate biological learning.
The recorded-circuit and randomized-circuit controls use the same input projection, degree sequence, number of parameters, experience and replay batches.

## Limits and provenance

- Reduced feedforward rate model, not the full fly brain or natural plasticity.
- One selected circuit and one rewiring; confidence intervals concern training seeds only.
- Fixed hyperparameters and update budget; no claim about optimally tuned algorithms.
- Parts A and B use different training regimes and state representations; compare agents within each part.
- Time-limit truncations are terminal in the existing maze; time remaining is not observed.
- Timing is cumulative update-call time, excluding data collection, initialization and evaluation.

Selected circuit: 256 Kenyon cells, 97 MBONs, 3,857 recorded positive connections. Reward training changed 3,013–3,283 permitted internal strengths per condition; missing connections stayed zero. The neural readout also trained. Neither model is the full-brain spiking simulator.

Scientific source: `f80cf6c780fc2783550d622e8fb15e8d8d626bfefdccce5024dccfec3f960f9e`. Run IDs: `20260921T060909Z-comparison-b15807`, `20260921T060903Z-comparison-a72cae`. Raw data/checkpoints remain under `runs/gx10-a/exp_2_comparison` and `runs/gx10-b/exp_2_comparison`, outside Git.

The first runner finalized `progress.json` after hashing it. Explicit `finalization_repair.json` audit records preserve the original manifests and exclude that mutable status pointer. All 378 immutable artifacts verified; no scientific output was changed.
