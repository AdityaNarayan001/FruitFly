# Full retained-connectome maze comparison

Full retained graph; engineered bounded recurrent rate model and two-tick TD gradients. Not biological physiology.

Tabular Q performed better on familiar success under this fixed budget. Tabular Q performed better on unseen success under this fixed budget.

Fixed before outcomes: five seeds, both reward choices, 10,000 shared transitions, one chronological pass, 50 familiar and 50 unseen frozen trials per condition. No cached neural responses. Activity resets only at trial boundaries.

| Controller | Familiar success | Unseen success | Mean training time | Trainable parameters |
| --- | ---: | ---: | ---: | ---: |
| q_table | 100.0% (500/500) | 51.8% (259/500) | 0.02 s | 10,368 |
| full_fixed | 10.0% (50/500) | 10.2% (51/500) | 42.69 s | 666,804 |
| full_plastic | 10.0% (50/500) | 10.2% (51/500) | 114.26 s | 25,136,216 |
| rewired_plastic | 10.0% (50/500) | 10.2% (51/500) | 217.28 s | 25,136,216 |
| random | 31.2% (156/500) | 28.4% (142/500) | 0.00 s | 0 |

## Paired uncertainty

Percentage-point differences, resampling five seed means after averaging reward A/B. These descriptive intervals have limited resolution.

- familiar: full_plastic minus q_table: -90.00 pp; 95% interval [-100.00, -80.00].
- unseen: full_plastic minus q_table: -41.60 pp; 95% interval [-46.40, -35.80].
- familiar: full_plastic minus full_fixed: 0.00 pp; 95% interval [0.00, 0.00].
- unseen: full_plastic minus full_fixed: 0.00 pp; 95% interval [0.00, 0.00].
- familiar: full_plastic minus rewired_plastic: 0.00 pp; 95% interval [0.00, 0.00].
- unseen: full_plastic minus rewired_plastic: 0.00 pp; 95% interval [0.00, 0.00].

## Post-hoc recurrence diagnostic

After partial primary outcomes, a separate frozen-policy ablation removed every recurrent connection while retaining the learned sensory readout. No retraining was done. 989/1000 complete paths and 995/1000 outcomes matched the original full-plastic controller. This is an explanatory post-hoc result, separate from the prospective comparison. The ablation used CPU float32 while the original used CUDA: numerical rounding may contribute to the small subset of differences, so those differences are not a pure causal estimate of recurrence.

## What was simulated and trained

Every decision advanced 166,700 nodes and 25,582,938 edge slots. Of these, 24,469,412 had nonzero signed initial weights; zero weights stayed zero. Engineered input drove 17,885 annotated sensory neurons.

- full_fixed: 0-0 internal edge strengths changed; 142,845-143,086 neurons exceeded the activity threshold during training. All nodes were simulated, including quiet ones.
- full_plastic: 8,383,190-8,688,335 internal edge strengths changed; 142,850-143,092 neurons exceeded the activity threshold during training. All nodes were simulated, including quiet ones.
- rewired_plastic: 22,386,313-22,819,364 internal edge strengths changed; 166,565-166,565 neurons exceeded the activity threshold during training. All nodes were simulated, including quiet ones.

The magnitude of the anatomical internal-weight change was 0.001691% to 0.002515% of the initial weight norm (L2 norm of the change divided by initial L2 norm). Count of changed edges alone does not establish a large functional change.
166,614 nodes are structurally reachable from driven inputs through nonzero modeled edges; 86 are not. Thresholded activity can be lower because of attenuation/cancellation.

All neural evaluations preserved parameter hashes; reloaded checkpoints reproduced the first familiar trial. Raw saved paths were replayed independently to verify each outcome.

## Limits

This is one artificial recurrent model, one fixed sensory projection, one fixed multigraph wiring null and a fixed training budget. The generic rate dynamics, sign normalization, privileged grid localization, four-action readout and two-tick truncated gradient rule are engineered. There is no validated natural sensory/motor mapping or dopamine mechanism. A weak result cannot rule out other dynamics, learning rules or budgets. Q-learning and the neural models are not separately hyperparameter-optimized. Equal data does not imply equal compute or parameter counts.

Familiar trials repeat one layout and two cue assignments, so trial counts are not independent training replications; the interval unit is the five seeds. The null preserves sign-stratified source degree counts and each target row weight multiset, but allows parallel pairs and self-edges. Unseen mazes vary walls, while start/food coordinates stay fixed. Food identity is hidden from the policy; raw visible cue assignment is supplied. Timeout transitions bootstrap before resetting. No test-result tuning was performed.

## Exact evidence

- gx10-a: `20260921T063920Z-fullscale-e4d345`, 32 immutable artifacts checked, 2500 trial trajectories replayed.
- gx10-b: `20260921T063951Z-fullscale-29e490`, 32 immutable artifacts checked, 2500 trial trajectories replayed.

Scientific source SHA-256: `1dd798bb860a282714893a48033d73346b47c9789ec8188ffe508e74217357e6`.
Graph SHA-256: `f2aa11508c3665389e7d511b2d68ebd5283fe763c60caecb022a85a09726f6e9`.
Raw data/checkpoints are outside Git in `runs/gx10-a/exp_2_fullscale` and `runs/gx10-b/exp_2_fullscale`. The source and protocol are versioned; the full external dataset is required to reproduce GPU behavior.
