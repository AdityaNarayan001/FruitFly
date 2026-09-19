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
