# FruitFlyBrain

A local, interactive workbench for connectome-based neural experiments. Draw an input, send a neural pulse, inspect model activity, and save an auditable session. It runs on a **CPU** or, on compatible Linux systems, **NVIDIA CUDA**.

**Scientific status:** exploratory software. Experiment 1 has no learning or validated motor decoder; Experiment 2 trains an external Q-learning controller while keeping neural weights fixed. The real-data input map and dynamics still need biological validation; strong pulses can produce unrealistic voltages. The default installation is a prominently labeled **synthetic 256-node teaching demo**, not the fly connectome.

## Start in three steps

1. Install **Python 3.11, 3.12 or 3.13**, including `venv` and `pip`.
2. Clone this repository (or download its ZIP).
3. Start the project:

```sh
git clone https://github.com/AdityaNarayan001/FruitFly.git
cd FruitFly
sh setup.sh
```

Open [the local dashboard](http://127.0.0.1:8765/). Keep the terminal open; **Ctrl+C stops the service**. Choose a pattern, then press **Run** or **Step**. No account, API key, NVIDIA GPU, SSH alias, Node.js installation or Git metadata is needed for the CPU demo. Internet is needed for first-time dependency installation.

Repository: [AdityaNarayan001/FruitFly](https://github.com/AdityaNarayan001/FruitFly). Setup works from the checkout and does not rely on the original developer's filesystem.

If another service uses port 8765:

```sh
sh setup.sh --port 8876
```

Then open `http://127.0.0.1:8876/`. The installer does not stop other services.

If Python is outside your PATH:

```sh
FFB_PYTHON=/path/to/python3.12 sh setup.sh
```

On some Linux distributions, install the OS package providing `python3-venv` first. Setup does not invoke `sudo`, install a package manager, or modify system Python. Unsupported Python/OS combinations fail with an actionable message instead of changing the pinned dependencies.

## Experiment 2: reward-guided maze learning

Experiment 1 remains available through `setup.sh` on port 8765. Its original code and PDFs are preserved at Git tag `exp-1-v0.2`.

```sh
sh setup_exp_2.sh
# Real graph on a supported NVIDIA Linux server:
sh setup_exp_2.sh --dataset male-cns --backend cuda
```

Open `http://127.0.0.1:8767/`. The default uses the labeled synthetic CPU graph. The same `--backend`, `--dataset`, `--port`, `--venv`, `--data-dir`, `--runs-dir` and `--no-start` options apply. Experiment 2 defaults to `runs/exp_2`; it never starts or stops Experiment 1.

The editable 9 x 9 arena has a fly avatar, a start and two food sites. Pattern A and B swap sites between trials. The agent must learn which pattern earns +1 and how to reach it; the wrong pattern gives -0.25, ordinary movement costs -0.01, and trials last at most 120 actions. Grid position, adjacent walls and both visible pattern beacons are engineered observations. This first version studies route learning, not unknown-maze search or realistic fly locomotion.

- **Train / Pause / Step:** train an external tabular Q controller in bounded batches; pacing changes display speed only.
- **Edit the maze:** place walls, start and food sites; Apply creates a new experiment with fresh learning and preserves the old record.
- **Manual control:** arrows move the avatar without training; reset the trial before switching between manual and autonomous modes.
- **Neural encoder:** two actual 40 ms reset probes stimulate mapped L1 columns; 32 measured spatial rate features feed a fixed template decoder. These deterministic probes are cached. Neural dynamics do not continuously advance as the avatar moves, and brain synapses do not learn.
- **Matched control:** the same Q learner reads the images directly and trains on matched trial seeds. Equal performance is expected when decoded information is identical.
- **Evaluate:** freezes Q and compares neural-cue Q, image-cue Q and random movement on familiar and unseen mazes. Unseen-layout performance measures the limits of route memory.
- **Save / Restore:** save learned Q tables and the completed-episode count. Restore creates a new linked run, starts a fresh trial and checks encoder identity. Partial-trial state is excluded.

The prospective protocol is `PLAN/exp_2.pdf`; its source was written before implementing learning. Records include source/graph/probe identities, commands, per-episode outcomes, frozen evaluation trials, source archives and checkpoints. No connectome advantage or biological learning follows from successful navigation alone.

For the existing lab deployment, run on `gx10-b` from `FruitFlyBrain/current`:

```sh
PYTHONPATH=src ../../.venv/bin/python -m fruitflybrain.exp2 \
  --graph ../../data/malecns-graph-v1 --runs ../../runs/exp_2 --backend cuda --port 8767
# On the client Mac:
ssh -N -L 8767:127.0.0.1:8767 gx10-b
```

## Use the actual MaleCNS graph

```sh
sh setup.sh --dataset male-cns --backend cpu
```

This downloads approximately 1.1 GB of pinned public data, verifies its checksums and prepares the graph. Plan for **at least 16 GB RAM, preferably 32 GB**, **6 GiB free disk** for preparation, and additional space for run records. Import peak memory is substantially larger than the prepared graph's approximately 406 MB disk size. Full-graph CPU simulation can be slow.

The imported graph retains **166,700 nodes, 25,582,938 directed edges and 124,177,617 synaptic contacts**. “Full retained graph” means all edges between annotations with nonempty superclass, including uncertain labels and isolated nodes; it does not mean all raw segments or complete physiology. The original contact counts remain separate from our assumed signed model weights.

The real dataset is downloaded only when you explicitly choose `--dataset male-cns`. Rerun the same command to reuse verified data. The default `sh setup.sh` always selects the demo; it does not silently remember a previous real-data selection.

## NVIDIA CUDA on Linux

Install a compatible NVIDIA driver and CUDA toolkit, including `nvcc`, then run:

```sh
sh setup.sh --dataset male-cns --backend cuda
```

The builder detects the first visible GPU's compute capability. `FFB_CUDA_ARCH=sm_89` is an example architecture override; use the value appropriate for your actual device. `FFB_NVCC=/path/to/nvcc` selects a compiler. The toolkit must support the selected architecture. GB10 is tested with CUDA 13.0 and `sm_121`; other GPUs require their own compatibility checks.

`--backend auto` selects CUDA when the Linux driver/compiler tools are discoverable, otherwise CPU. An explicit CUDA request does not silently fall back after a failure. **Macs use CPU mode; this project has no Metal/MPS backend.**

| Platform | Route | Verification status |
| --- | --- | --- |
| Apple Silicon Mac | CPU demo and real graph | Tested locally |
| Linux ARM64 / ASUS GX10 | CPU and native CUDA | Tested on the project hosts |
| Intel Mac / Linux x86-64 | CPU if pinned wheels support the OS | Intended compatible, not hardware-tested here |
| Windows | Suitable WSL2 Linux environment | Intended route; native Windows setup is not validated |
| Phone / tablet | Browser client through a trusted private tunnel | Does not execute the simulator locally |
| Other devices | Compatible Python/OS, wheels and enough memory required | No universal-device guarantee |

## Setup options and generated files

```sh
sh setup.sh --no-start
sh setup.sh --backend cpu --dataset demo --port 8876
sh setup.sh --venv /path/to/venv --data-dir /path/to/data --runs-dir /path/to/runs
```

`--no-start` installs, prepares data and runs the numerical doctor check, then prints a start command. Repeated setup reuses a matching dependency environment. It preserves existing runs and refuses conflicting dataset paths.

- `.venv/`: isolated dependencies: NumPy 2.2.6, PyArrow 19.0.1 and SciPy 1.15.3.
- `data/demo-v1/`: deterministic invented teaching graph.
- `data/malecns-v1/`: raw source files and download metadata when requested.
- `data/malecns-graph-v1/`: prepared real graph, annotations and integrity metadata.
- `build/`: optional CUDA library and compiler/build record.
- `runs/local/`: local sessions; `_sources/` keeps one source archive per code identity.

Stop and restart the service after editing code. A running process already imported its code; a consistent source archive is required before recording a new session.

## What the interface shows

- **Input:** blank, uniform, grating, spot or hand-painted 32 x 24 image; eye selection, Drive, contrast, frequency, spatial cycles and spot radius. The actual displayed grid is sampled for direct L1 drive. Both eyes receive the same image through independently normalized coordinate maps.
- **Inside the circuit:** rotate, zoom and filter sampled landmarks colored by measured model activity. Real mode displays 5,028 sampled landmarks while all 166,700 retained cells continue to simulate. Demo coordinates and labels are explicitly synthetic.
- **Response:** whole-network mean and left/right L1, T4, T5 and descending-population rates; history and a sampled spike raster. Soma side is not a natural steering command.
- **Manual pulses:** inject signed increments into a selected group or individual neuron; pulse duration plus a 100 ms observation interval advances automatically within the session limit.
- **Inspector:** search type/body ID, inspect latest voltage and spikes, and list strong modeled incoming connections.
- **Controls:** Run for a bounded window, Pause at a chunk boundary, Step 20 ms, Reset or Finish & save. Sessions automatically finish at 10 seconds simulated time. Space and arrow shortcuts operate outside form fields.
- **Saved View:** review recorded telemetry with control disabled. It is not full historical-state restoration.

The real image encoder targets **1,767 of 1,776 L1 cells** with explicit column coordinates. It is **not calibrated fly vision**. Experiment 1 has no photoreceptor-transduction model, validated motor behavior, reward mechanism, language model or training loop. Experiment 2 adds the separate external controller described above.

## Reproducibility and tests

Each interactive session saves a manifest, source identity/archive, initial input, tick-stamped commands, graph/mapping identities, input digest, per-chunk summaries, sampled raster and all-neuron total spike counts. It does not retain a complete full-neuron voltage/spike trajectory or a resumable neural checkpoint. Preserve the source archive, graph and original records when reproducing a run; cross-backend equality must be checked.

From the project root:

```sh
.venv/bin/python -m fruitflybrain.cli doctor --backend cpu
.venv/bin/python -m unittest discover -s tests -v
```

On a prepared CUDA host:

```sh
FFB_TEST_CUDA=1 .venv/bin/python -m unittest discover -s tests -v
```

The suite checks causal edge direction/delay, reference/CPU/CUDA trajectories, chunk boundaries, refractory behavior, reset, exact IDs, import integrity, eye-specific input, pulse timing, saved-session completion, control origin/session checks, and synthetic-data labeling. The CUDA test is explicitly skipped without `FFB_TEST_CUDA=1`.

Optional browser interaction checks require Node.js, Playwright and Chrome. They run actual exploratory sessions against the selected service:

```sh
FFB_BASE_URL=http://127.0.0.1:8876 node tests/browser_smoke.cjs
```

Use `FFB_PLAYWRIGHT=/path/to/playwright` or `FFB_CHROME=/path/to/chrome` if needed. These packages are not necessary to operate the dashboard. The check creates `work/ui-qa` for its artifacts.

## Documents and data examples

- `PLAN/plan.pdf`: the comprehensive beginner handbook, with real dataset examples, neuroscience vocabulary, model equations, every control, worked exercises, setup, troubleshooting, glossary and references.
- `PLAN/exp_1.pdf`: prospective visual-motion protocol and dated engineering amendment.
- `PLAN/exp_2.pdf`: prospective reward-choice/maze protocol, learning rule, controls and checkpoint semantics.
- `docs/research/`: editable document sources and builders. `evidence/dataset_samples.json` contains exact sample rows, schemas and source hashes.
- `STATUS.md`: observed results, implementation milestones and outstanding scientific gates.

Exactly one current copy of each PDF lives in `PLAN`. Rebuilding documents is separate from runtime installation and requires ReportLab:

```sh
python docs/research/build_handbook.py       # plan.pdf only
python docs/research/build_pdfs.py           # both current documents
```

Render and visually review changed PDFs before delivering them. Research sources and PDFs are excluded from GPU releases.

## Optional remote execution

For your own server, install this checkout there with `setup.sh`. Keep its service running, and forward its loopback port from the client:

```sh
ssh -N -L 8876:127.0.0.1:8765 your-server
```

Then open `http://127.0.0.1:8876/` on that client. The browser renders on the client while the server computes. A phone's `127.0.0.1` is the phone itself, so mobile clients need their own trusted access/tunnel arrangement. The service is loopback-bound and is not a hardened public multiuser deployment.

The original lab has SSH aliases `gx10-a` and `gx10-b`. Its optional source-manifest workflow is:

```sh
python3 scripts/remote.py deploy gx10-a
python3 scripts/remote.py deploy gx10-b
python3 scripts/remote.py bootstrap gx10-a
python3 scripts/remote.py bootstrap gx10-b
python3 scripts/remote.py collect gx10-a
python3 scripts/remote.py collect gx10-b
```

Those aliases are not required for other users. Deployments create immutable hash-named releases and preserve snapshots; upload staging is temporary. Data and runs are never deleted by deployment. The two machines run independent processes; distributed multi-GPU simulation is not implemented.

For the lab's existing graph/service layout, start from its remote `FruitFlyBrain/current`:

```sh
PYTHONPATH=src ../../.venv/bin/python -m fruitflybrain.cli serve \
  --runs ../../runs --graph ../../data/malecns-graph-v1 --backend cuda --port 8765
```

Legacy batch-engineering configs under `configs/` are retained for the original remote layout; use the portable setup commands above for ordinary local operation. CLI `run` only permits engineering mode until the scientific protocol is ready.

## Sources, scope and next scientific work

MaleCNS is a collaboration involving Janelia/FlyEM, Cambridge, MRC-LMB and Google Research. See the [official dataset](https://male-cns.janelia.org/download/), [Cell paper](https://doi.org/10.1016/j.cell.2026.08.015) and [Google overview](https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/). Dataset CC-BY terms and dependency licenses remain separate. A project software license has not yet been selected; publishing the repository does not itself grant a software license.

Before scientific visual-motion evaluation: validate mapping and model stability, define evidence-backed readout populations, implement the specified connectivity control, and record a dated freeze amendment. Experiment 2 has its own prospective protocol and does not remove Experiment 1's scientific gates. Numerical tests and a working UI do not establish biological validity.
