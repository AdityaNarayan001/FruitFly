# FruitFlyBrain

A local, interactive workbench for connectome-based neural experiments. Draw an input, send a neural pulse, inspect model activity, and save an auditable session. It runs on a **CPU** or, on compatible Linux systems, **NVIDIA CUDA**.

**Scientific status:** exploratory software. Experiment 1 has no learning or validated motor decoder; Experiment 2 trains an external Q-learning controller while keeping neural weights fixed. The real-data input map and dynamics still need biological validation; strong pulses can produce unrealistic voltages. The default installation uses the **real MaleCNS connectome** and preserves Experiment 1's rotatable brain landmark view.

## Clone once and choose a dataset

Install **Python 3.11, 3.12 or 3.13** with `venv` and `pip`, then:

```sh
git clone https://github.com/AdityaNarayan001/FruitFly.git
cd FruitFly
```

All commands below run from this folder. Each launcher creates or reuses `.venv`, installs pinned dependencies and prepares the selected dataset. Keep its terminal open; **Ctrl+C stops that service**. Open the printed URL manually in your browser.

**Plain `sh setup.sh` starts with the real fly connectome and its 3D brain landmarks.** All three launchers default to real data; the small synthetic circuit is available only when explicitly requested:

| Dataset | What you get | Option |
| --- | --- | --- |
| MaleCNS (default) | 166,700 retained neurons from the real connectome; annotated brain/CNS landmarks | No option needed, or `--dataset male-cns` |
| Optional demo | 256 invented neurons in a labeled schematic grid; quick CPU test | `--dataset demo` |

Real-data preparation downloads about **1.1 GB** and needs **at least 16 GB RAM (32 GB recommended)** and **6 GiB free disk**, plus space for records. Full-graph CPU simulation can be slow. The anatomy explorer only reads the graph and does not run neural dynamics.

## Choose an interface

These quick-start commands use the **real connectome**. The first run downloads/prepares it; subsequent runs reuse it. Add `--dataset demo` only for a small synthetic software test.

| Interface | Start command | Browser address | Purpose |
| --- | --- | --- | --- |
| Experiment 1 | `sh setup.sh` | [localhost:8765](http://127.0.0.1:8765/) | Apply visual input / pulses and inspect simulated neural activity |
| Experiment 2 | `sh setup_exp_2.sh` | [localhost:8767](http://127.0.0.1:8767/) | Train and evaluate an external reward-learning maze controller |
| Neural anatomy explorer | `sh setup_explorer.sh` | [localhost:8768/network](http://127.0.0.1:8768/network) | Inspect annotated populations, neuron locations and directed connections |

Use a **separate terminal for each service** if you want all three open. They reuse the environment and prepared dataset. Stop only the terminal belonging to the service you want to stop. No account, API key, SSH alias, Node.js installation or NVIDIA GPU is required for CPU mode; internet is needed for first-time installation and any requested data download.

## Experiment 1: visual input and neural response

For the **actual connectome** on a Mac or other supported CPU machine:

```sh
sh setup.sh --dataset male-cns --backend cpu
```

For a supported NVIDIA Linux server with its driver and CUDA toolkit installed:

```sh
sh setup.sh --dataset male-cns --backend cuda
```

For the small teaching demo:

```sh
sh setup.sh --dataset demo
```

Open [Experiment 1](http://127.0.0.1:8765/). Select a visual pattern and press **Run** or **Step 20 ms**. You can paint an input, select an eye, send a pulse, inspect neurons and save the session. Learning is off. In real-data mode, drag the brain landmark view to rotate it and scroll to zoom. Demo mode shows labeled node groups; their placement is a schematic, not a brain shape.

The original Experiment 1 baseline is retained at Git tag `exp-1-v0.2`. Subsequent UI clarifications do not change its neural dynamics or scientific protocol.

## Experiment 2: reward-guided maze learning

For the real connectome on CPU, or the quick demo:

```sh
sh setup_exp_2.sh --dataset male-cns --backend cpu
# Or the small synthetic demo:
sh setup_exp_2.sh --dataset demo
```

On a supported NVIDIA Linux server:

```sh
sh setup_exp_2.sh --dataset male-cns --backend cuda
```

Open [Experiment 2](http://127.0.0.1:8767/). It defaults to `runs/exp_2` and starts independently of Experiment 1. Initial loading measures two fixed 40 ms neural probes before the maze becomes ready.

1. Keep the default maze or place walls, the start and two food sites. **Apply new experiment** saves the previous record and starts fresh learning.
2. Choose a batch size and press **Train**. Use **Pause**, **Step** and the pacing slider to inspect decisions.
3. To move manually, pause and reset the trial first. Manual moves do not train the controller or enter its learning metrics.
4. Pause and select **Evaluate frozen policies** to compare the learned controller, image-only control and random movement on familiar and unseen mazes.
5. **Save checkpoint** stores completed-episode learning. **Restore as new experiment** creates a linked run and preserves the old record.

Pattern A and B swap food sites between trials. Correct food gives +1, the other gives -0.25, ordinary movement costs -0.01, and a trial lasts at most 120 actions. Grid position, adjacent walls and both pattern beacons are engineered observations. This is route learning with visible cues, not an unknown-maze search model.

**What learns:** an external tabular Q controller. **What stays fixed:** the connectome, neuron dynamics and calibrated decoder. The two measured neural responses are cached; brain activity does not advance at every maze step. The matched image-cue controller trains with the same trial seeds. Equal performance does not establish a connectome advantage.

The prospective protocol is `PLAN/exp_2.pdf`. Records contain source/graph/probe identities, commands, episode outcomes, frozen evaluation trials, source archives and checkpoints. Partial-trial state is excluded from checkpoints.

## Neural anatomy explorer: inputs, circuits and outputs

For the **actual annotated brain/CNS**, without starting either experiment:

```sh
sh setup_explorer.sh --dataset male-cns
```

For the small synthetic teaching graph:

```sh
sh setup_explorer.sh --dataset demo
```

Open [the neural explorer](http://127.0.0.1:8768/network). No CUDA backend is needed: the service reads anatomy on the server CPU and the browser renders the view. It never stimulates neurons, trains a policy or creates an experiment run.

Experiment 2 also includes the same explorer: click **Explore neural anatomy**, or open [localhost:8767/network](http://127.0.0.1:8767/network) while a current Experiment 2 service is running. This route shares its loaded dataset; no second launcher is needed.

- **Anatomical landmarks:** drag to rotate, use the zoom buttons, and filter populations. Counts cover the full graph; the view samples located cells.
- **Selected connections:** shows incoming partners, the selected neuron and outgoing partners in a schematic layout. Reciprocal partners can appear on both sides. This is a diagram, not a physical neuron shape.
- **Experiment 2 input / readout:** pink rings mark selected L1 cells used for both roles. This choice is separate from biological sensory and motor categories.
- **Search / click:** inspect exact neuron IDs, types such as L1/T4, or superclasses such as descending_neuron. Search includes cells absent from the display sample.
- **Directed connections:** up to 20 incoming and 20 outgoing partners, ranked by recorded contact counts when available. Zero assumed model weight does not erase an anatomical connection.

The retained MaleCNS annotations include 815 `cb_motor`/`vnc_motor` neurons and 1,314 `descending_neuron` cells. These are distinct categories. Neither population controls the Exp 2 avatar. Processing is distributed across circuits; there is no fixed number of ANN hidden layers. Artificial recurrent networks also have feedback.

Missing coordinates are never invented. Anatomical lines join cell landmarks, not axon paths or synapse locations. Colors identify annotation groups, not live activity or verified task computations. See the [MaleCNS annotations](https://male-cns.janelia.org/) and [descending pathway description](https://www.janelia.org/project-team/fly-descending-interneuron).

**Biological limitation:** L1 names a lamina neuron type, not layer 1. Biological L1 uses graded voltage responses; our uniform spiking model is not a faithful L1 physiology model. [L1 physiology paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC3806040/).

## If the viewer shows parallel slashes or a grid

In older versions, the default demo's invented 3D grid was viewed nearly edge-on, producing small parallel slashes. This was the **256-node synthetic fixture**, not a corrupted download or a missing brain mesh. Current Experiment 1 displays that fixture as a clearly labeled schematic with individually clickable nodes.

To update and display the actual brain landmarks:

```sh
# Stop the old service with Ctrl+C in its terminal, then:
git pull --ff-only
sh setup.sh --dataset male-cns --backend cpu
```

Refresh the browser after restarting. Check that the badge says **MaleCNS** and the count is **166,700 modeled neurons**. If it says **SYNTHETIC DEMO** and **256**, you are still running the demo. The real viewer shows sampled neuron landmarks, not a solid brain surface or complete neuron meshes.

The current default `sh setup.sh` selects the real connectome, even if an older installation already has demo data. Experiment 2 and the standalone explorer use the same real-data default. `--dataset demo` is an explicit opt-in; setup never substitutes it if real-data preparation fails. Existing verified data is reused; there is no need to delete it or saved runs.

## Ports, Python and setup troubleshooting

If a port is busy, stop your old instance or choose another port. Setup never kills another service:

```sh
sh setup.sh --port 8876
sh setup_exp_2.sh --port 8877
sh setup_explorer.sh --port 8878
```

Use the corresponding port in the browser URL. The explorer still uses `/network`.

If Python is outside PATH:

```sh
FFB_PYTHON=/path/to/python3.12 sh setup.sh
# The same override works with setup_exp_2.sh and setup_explorer.sh.
```

On some Linux distributions, install the OS package providing `python3-venv` first. Setup does not invoke `sudo`, install a package manager, or modify system Python. Unsupported Python/OS combinations fail with an actionable message. Windows users should use a suitable WSL2 Linux environment; native Windows setup is not validated.

## Real-data scope and storage

The importer retains **166,700 nodes, 25,582,938 directed edges and 124,177,617 synaptic contacts**. It keeps annotations with nonempty superclass, including uncertain labels and isolated nodes. This does not include every raw segment or complete physiology. Original contact counts are stored separately from our assumed signed model weights.

The prepared graph occupies approximately 406 MB, but download/import peak memory is substantially higher. The 6 GiB preparation allowance and RAM guidance above include working overhead. Data and run files stay outside Git.

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

All three launchers support `--no-start`, `--dataset`, `--port`, `--venv` and `--data-dir`. Experiment launchers also accept `--backend` and `--runs-dir`. `--no-start` installs, prepares data and runs the numerical doctor check, then prints the appropriate start command. Repeated setup reuses a matching dependency environment. It preserves existing runs and refuses conflicting dataset paths.

- `.venv/`: isolated dependencies: NumPy 2.2.6, PyArrow 19.0.1 and SciPy 1.15.3.
- `data/demo-v1/`: deterministic invented teaching graph.
- `data/malecns-v1/`: raw source files and download metadata when requested.
- `data/malecns-graph-v1/`: prepared real graph, annotations and integrity metadata.
- `build/`: optional CUDA library and compiler/build record.
- `runs/local/`: Experiment 1 sessions; `_sources/` keeps one source archive per code identity.
- `runs/exp_2/`: Experiment 2 records and learned-controller checkpoints.
- The read-only explorer does not create a runs directory.

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
python docs/research/build_pdfs.py           # configured research PDFs
```

Render and visually review changed PDFs before delivering them. Research sources and PDFs are excluded from GPU releases.

## Optional remote execution

For your own server, install this checkout there and start the desired interface with its launcher above. Keep that service running and forward its loopback port from the client. For Experiment 1:

```sh
ssh -N -L 8876:127.0.0.1:8765 your-server
```

Then open `http://127.0.0.1:8876/` on that client. For Experiment 2 use `ssh -N -L 8767:127.0.0.1:8767 your-server`; for the standalone explorer use `ssh -N -L 8768:127.0.0.1:8768 your-server`, then open `/network` on port 8768. The browser renders on the client while the server computes. A phone's `127.0.0.1` is the phone itself, so mobile clients need their own trusted access/tunnel arrangement. The service is loopback-bound and is not a hardened public multiuser deployment.

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

For the lab's existing shared graph/environment layout, first bootstrap the deployed release as above. Run the following commands from remote `FruitFlyBrain/current`, each in its own terminal:

```sh
# Experiment 1, normally gx10-a:
PYTHONPATH=src ../../.venv/bin/python -m fruitflybrain.cli serve \
  --runs ../../runs --graph ../../data/malecns-graph-v1 --backend cuda --port 8765
# Experiment 2, normally gx10-b:
PYTHONPATH=src ../../.venv/bin/python -m fruitflybrain.exp2 \
  --runs ../../runs/exp_2 --graph ../../data/malecns-graph-v1 --backend cuda --port 8767
# Independent read-only explorer, normally gx10-b:
PYTHONPATH=src ../../.venv/bin/python -m fruitflybrain.explorer \
  --graph ../../data/malecns-graph-v1 --port 8768
```

Legacy batch-engineering configs under `configs/` are retained for the original remote layout; use the portable setup commands above for ordinary local operation. CLI `run` only permits engineering mode until the scientific protocol is ready.

## Sources, scope and next scientific work

MaleCNS is a collaboration involving Janelia/FlyEM, Cambridge, MRC-LMB and Google Research. See the [official dataset](https://male-cns.janelia.org/download/), [Cell paper](https://doi.org/10.1016/j.cell.2026.08.015) and [Google overview](https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/). Dataset CC-BY terms and dependency licenses remain separate. A project software license has not yet been selected; publishing the repository does not itself grant a software license.

Before scientific visual-motion evaluation: validate mapping and model stability, define evidence-backed readout populations, implement the specified connectivity control, and record a dated freeze amendment. Experiment 2 has its own prospective protocol and does not remove Experiment 1's scientific gates. Numerical tests and a working UI do not establish biological validity.
