# Project status

Updated 2026-09-19. Phase: interactive engineering workbench. No visual-motion or learning result yet.

- Completed: research overview and prospective Experiment 1 PDFs, rendered and visually reviewed.
- Completed: local source layout, immutable hash-based deployment and preserved source snapshots; Git publication authorized to AdityaNarayan001/FruitFly.
- Completed: isolated pinned Python environments and native CUDA compilation on both GX10s.
- Completed: initial ten checks passed on each GX10, including CPU/CUDA trajectories across multiple delays and chunk boundaries, exact integer IDs, file integrity and ordered Arrow annotations.
- Completed: real-data import on gx10-a: 166,700 retained neurons, 25,582,938 directed edges, 124,177,617 contacts and 217 isolated neurons. 1,113,526 anatomical edges have zero effective fast weight under the declared transmitter policy.
- Completed: initial full-graph CUDA engineering run `20260919T170512Z-f2f23101`: 200 ms simulated in 0.581 s execution time (0.943 s including setup/telemetry). This short benchmark is approximately 0.34x real time, is not an optimized steady-state speed claim, and does not test visual motion. Source snapshot: `f7eed6e2360e98eebfba05b22385e9b208760f22bbdc2bf7a0b6d69137f5a454`.
- Completed: eight-neuron engineering fixtures on both hosts produced identical input digests and 932 spikes each; results synchronized under `runs/gx10-a` and `runs/gx10-b` on the Mac.
- Completed: read-only remote dashboard checked in the Mac browser through an SSH tunnel. Run selection, graph counts, rate plots and source identity display actual archived telemetry.
- Completed: interactive GPU backend and Mac UI with patterned input, eye/drive controls, manual group or neuron pulses, exact-step/pause/reset/save controls, rotatable sampled anatomy, neuron inspector, population charts and recorded spike raster. L1 column input remains exploratory.
- Verified: 16 backend tests on gx10-a, including input sampling, pulse timing, command records and control-session checks; 12 live browser interaction checks passed. Final deployment verification is recorded under work/ui-qa.
- Observed limitation: development run `20260919T172706Z-interactive-c6d167` reached a mean voltage of approximately -419.22 mV at 620 ms after a T4a pulse. Raw evidence is preserved; the model is not biologically validated and needs stability work before interpreting circuit behavior.
- Completed: Experiment 1 PDF updated in place to v0.2 (9 pages), retaining the original prospective protocol and adding a dated engineering amendment after development observations.
- Completed: portable SciPy CPU backend, default synthetic teaching demo and explicit real-data setup. `setup.sh` creates an isolated environment, verifies dynamics, prepares the chosen data and launches the UI without personal paths or SSH aliases.
- Verified: fresh and repeated Mac installation from a path with spaces and another working directory; 12 browser checks on the CPU demo. Full retained graph ran on the Apple M5 Pro CPU: run `20260919T175111Z-be16621d`, 100 ms simulated in 1.640 s execution (0.061x), 263,861 spikes. This is a short engineering benchmark, not biological validation or a controlled speed comparison.
- Completed: local run source archives, truthful backend/dataset labels and completion published only after session files finish saving.
- Completed: plan.pdf expanded to handbook v1.0 with exact source-table examples, all interface controls, portable installation and scientific limitations; its evidence and builders remain in docs/research.
- Pending scientific gates: validated retinal mapping, independently assigned direction/readout populations, model stability/sensitivity, validated rewiring controls, and a dated freeze amendment before evaluation.

The current run command intentionally permits engineering mode only. Synthetic fixtures and direct-stimulation benchmarks are not evidence of optomotor behavior. Current PDFs live only in `PLAN`; editable sources and revision notes live in `docs/research`. Per the user's layout correction, duplicate PDF exports and persistent local deployment staging were removed. One source archive used by completed runs remains under `runs/sources` for reproduction.

The initial failed import's partial graph is retained remotely as `data/malecns-graph-v1-incomplete-3010f35` for audit; it is not referenced by any accepted run. Subsequent importer versions publish output only after complete construction. The interactive service keeps its graph loaded on gx10-a and starts paused; computation advances only for requested bounded runs or pulses. gx10-b remains available for future matched controls.

## Experiment 2 (20 September 2026)

- Prospective exp_2.pdf written before implementing the learning rule; 11 pages, rendered and inspected. Experiment 1 code/UI/dynamics and both original PDFs are preserved; Git tag exp-1-v0.2 identifies the baseline.
- Implemented: independent launcher and service, editable maze and food sites, swapped pattern assignments, fixed cached neural probes, explicit external Q-learning, matched image-cue learner, random baseline, familiar/unseen frozen evaluation, boundary checkpoints and linked restoration.
- Synthetic development check: seed 42, 400 training trials, 30/30 familiar-maze successes with shortest successful paths for either rewarded pattern. The matched image-cue learner has identical Q values; this is not a connectome advantage or a biological-learning result. Real-data and UI verification are recorded in the experiment run artifacts.
- Real MaleCNS/CUDA exploratory run `20260920T093545Z-exp2-86f29e`: 400 training episodes, seed 42. Frozen familiar evaluation: neural-cue and image-cue learners both 30/30 successes, mean 11 actions and 100% successful-route efficiency; random 11/30. Unseen layouts: both learners 8/30, random 9/30. Weights remained frozen. This supports task-specific route learning and reveals poor transfer; no connectome advantage. Full-graph probe voltage minima were approximately -143 and -157 mV: physiology remains unvalidated despite finite responses.

## Read-only neural anatomy explorer (20 September 2026)

The explorer adds annotated landmark positions, population counts, selected experimental L1 inputs/readouts, ID/type search and directed anatomical neighbors. It does not assign a natural motor decoder or change either experiment's neural dynamics, learning rule or checkpoint. Unknown positions are not fabricated; visual sampling is explicit. Biological L1 uses graded responses, so our uniform spiking approximation remains unvalidated for that cell type. The lab explorer can run separately on gx10-b/8768 to preserve ongoing experiment sessions; newly started Exp 2 services also offer /network.

## Fresh-install viewer clarification (20 September 2026)

Reproduced the reported parallel slashes using the default setup.sh CPU demo. They were invented grid coordinates projected nearly edge-on, not missing MaleCNS data. The demo now uses an explicitly labeled 2D grouping by synthetic cell type and side, retaining node IDs and measured activity; real-data projection and all neuron dynamics are unchanged. The README now separates all three launch paths, demo versus real data, expected visuals, resource requirements and update steps. setup_explorer.sh reuses the existing installer and launches only the read-only explorer. The Experiment 1 UI correction is authorized by the user's report; the original baseline remains tagged.

## Real connectome default (20 September 2026)

Following the user's preference for the original Experiment 1 brain view, all three launchers now default to MaleCNS. Plain setup.sh prepares the real dataset and displays its existing rotatable 3D brain landmarks; the grouped synthetic view requires an explicit --dataset demo. Real-data preparation failures do not fall back to demo data. README commands and first-run resource requirements reflect this default. Rendering, neural dynamics, saved runs and the external learning rule are unchanged.

## Matched maze comparisons (21 September 2026)

- Added a separate comparison runner and a prospective v0.2 PDF amendment before the new controller was implemented/evaluated. Existing live experiments and original learning mechanisms are preserved. A dated v0.3 results section follows the completed runs.
- Original-controller audit, gx10-a run `20260921T060909Z-comparison-b15807`: ten seeds, both rewarded patterns, 400 online episodes each. Q tables were identical after every episode. Both controllers: 2,000/2,000 familiar successes and 570/2,000 unseen successes. No benefit from the cached neural probes.
- Reduced-circuit run, gx10-b `20260921T060903Z-comparison-a72cae`: 256 sampled Kenyon cells, 97 MBONs, 3,857 recorded positive connections. Engineered rate dynamics/projection/readout; not full-brain LIF or validated biological plasticity. Reward changed 3,013-3,283 permitted internal strengths per condition. Frozen and absent connections verified.
- Shared experience: 20,000 transitions, 2,000 replay batches of 64, ten seeds, both reward assignments. Final familiar/unseen success: Q table 85.0%/52.3%; frozen circuit plus trained readout 37.5%/19.6%; plastic anatomical circuit 87.5%/24.9%; degree-matched rewired plastic circuit 95.0%/26.9%; random 31.3%/29.05%. Each agent/split has 2,000 evaluation trials.
- Plastic minus Q: familiar +2.5 percentage points (paired seed-bootstrap 95% interval -12.5 to +17.5), unseen -27.4 (-35.7 to -18.05). No established familiar advantage; Q transferred better in this setting. No demonstrated anatomical benefit over the matched rewired network. One sampled circuit, fixed untuned hyperparameters and finite-horizon limitations constrain interpretation.
- Scientific source SHA-256: `f80cf6c780fc2783550d622e8fb15e8d8d626bfefdccce5024dccfec3f960f9e`. Source archive, raw shared experience, checkpoints, all trials and manifests are preserved outside Git. Compact derived results are in `docs/research/maze_comparison_results.md` and JSON. The first runner hashed progress before its final update; explicit audit records preserve original manifests and verify all 378 immutable artifacts. No scientific output was changed by that bookkeeping repair.
