# FruitFlyBrain project instructions

The user wants the authoritative code on this Mac, with computational experiments running on `gx10-a` and `gx10-b`, and a Mac browser interface. Keep code synchronized using the source-manifest deployment scripts. Preserve source archives and per-run manifests. The user selected `git@github.com:AdityaNarayan001/FruitFly.git` as the Git remote. Keep data, environments, generated builds and run records out of Git.

Maintain exactly one PDF folder on the Mac: `FruitFlyBrain/PLAN`, containing `plan.pdf` and `exp_*.pdf`. Do not copy these PDFs into Codex outputs, deployment folders or GPU releases. Editable document sources, the builder and revision notes belong in `docs/research`. Record revisions and whether results had been observed, without automatically creating redundant PDF copies. Write `exp_2.pdf` before adding a concrete reward-learning mechanism. This is a documentation workflow, not a request to schedule autonomous future work. Keep upload staging temporary and delete it automatically after deployment; do not recreate `work/deploy`.

Distinguish anatomy, predicted labels, assumed dynamics, engineered encoders/decoders and trainable parameters. Cite primary sources for scientific claims. Never present a synthetic fixture, direct-stimulation benchmark, UI animation or trained external controller as evidence that the fly itself learned a task. Keep negative results and exact experimental deviations.

Experiment 1 evaluation is gated on audited retinal/readout mappings, model stability, a tested connectivity null and a dated freeze amendment. Engineering work and development runs may proceed within the user's authorization. Never invent retinal positions or natural motor functions to make a demo work.

Do not alter unrelated remote projects or services. Remote environments live under `~/FruitFlyBrain`; source deployments must not delete data or runs. Keep the dashboard loopback-bound and access it through SSH forwarding. Use the PDF skill's render-and-inspect workflow for document changes.
