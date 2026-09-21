# Research document series

| Document | Version | Date | Status |
| --- | --- | --- | --- |
| plan.pdf | 1.0 | 2026-09-19 | Beginner handbook: biology, data examples, full UI reference, portable setup and research roadmap |
| exp_2.pdf | 0.3 | 2026-09-21 | Original protocol, prospective comparison amendment, and dated observed results |
| exp_1.pdf | 0.2 | 2026-09-19 | Prospective visual-motion protocol plus dated interactive engineering amendment; evaluation remains gated |

Editable content is in this directory's JSON files. From the project root, rebuild with `python docs/research/build_pdfs.py` using ReportLab. Output goes only to `PLAN/plan.pdf` and `PLAN/exp_*.pdf`. These are local prospective documents, not external preregistrations. The project repository is https://github.com/AdityaNarayan001/FruitFly. Statements in the dated PDFs about Git being deferred describe their preparation before initial publication.

Record revisions here, including whether relevant results had been observed. Keep one current PDF per document; do not create duplicate export or deployment copies. Preserve experimental evidence in run records. Publish dated mapping/model amendments before confirmatory trials. Add `exp_2.pdf` before a concrete reward-learning experiment begins. Do not fabricate future results or overwrite unfavorable findings.

2026-09-19 layout correction: PDF contents unchanged. Removed duplicate exports and persistent upload staging. Moved editable sources out of PLAN. This layout policy supersedes the initial PDF's proposed automatic archive/deployment layout.

2026-09-19 v0.2 amendment: exp_1.pdf is now 9 pages. Added the interactive workbench, exact input/display coverage, run logging and an observed nonphysiological voltage response. The original protocol remains prospective; the added engineering observations are explicitly retrospective. Rendered all nine pages and visually reviewed the changed first page, page numbering and amendment. plan.pdf remains unchanged.

2026-09-19 handbook v1.0: replaced the initial overview with a book-length tutorial and reference. Added exact verified dataset samples, model equations, all dashboard controls, worked exercises, CPU/CUDA setup, troubleshooting, glossary, references, version-specific community-project review and observed Mac CPU performance. Written after engineering runs; no confirmatory behavioral or learning result claimed. The plan builder is build_handbook.py; build_pdfs.py dispatches to it for plan.json. All current PDFs remain only in PLAN.

2026-09-19 final handbook verification: 64 pages rendered and visually inspected; clickable contents and references checked. Fresh-checkout setup passed on Mac CPU and Linux ARM64 CPU/CUDA. Both backends passed 13 browser control checks, including delayed Reset acknowledgement. Numerical/API suites passed 19 tests on Mac plus one expected CUDA skip, and all 20 tests on both GX10s. Existing exp_1.pdf preserved.

2026-09-19 initial Git publication: source, current PDFs and editable research sources are included. Downloaded datasets, run records, environments and build outputs remain excluded. The handbook and Experiment 1 PDFs retain their dated pre-publication provenance.

2026-09-20 Experiment 2: an 11-page prospective protocol was generated before implementation or learning outcomes. Rendered and inspected all pages. Original plan.pdf and exp_1.pdf bytes are preserved. Rebuild only the new document with `python docs/research/build_pdfs.py exp_2.json`; the generic builder now supports a document-specific date and an explicit file selection. Development results remain in STATUS.md and run records.

2026-09-21 v0.2: added the matched comparison protocol before implementing or observing the new plastic-network benchmark. Original Exp 2 results were known and are explicitly identified. Real-data default setup correction is dated in the amendment. No full-brain training or biological learning claim.

2026-09-21 v0.3: after completing both ten-seed comparisons, appended observed results, the familiar/unseen chart, paired intervals, parameter-change evidence and limitations. No anatomy advantage was demonstrated. The original v0.2 prospective PDF is preserved by Git commit f9b49bb; one current exp_2.pdf remains in PLAN. Raw-run hashes verified with an explicit audit correction for the mutable progress pointer.

- 21 September 2026, v0.4: Exp 2 gained a prospective full-retained-connectome
  amendment (pages 20-22), before full-scale implementation and maze results.
  Prior reduced-circuit outcomes were already known. Configuration stores the
  exact prospective PDF hash. Continuous bounded recurrence and truncated
  two-tick TD gradients are engineered assumptions, not validated physiology.
- 21 September 2026, after partial full-scale outcomes: a separate post-hoc
  zero-recurrence diagnostic was defined in `configs/exp_2_fullscale_diagnostic.json`.
  It will retain saved sensory mapping/readout parameters and remove recurrent
  strengths, with no retraining or changes to the primary protocol. All ten saved
  plastic conditions and all primary trial definitions are included. This is an
  explanatory ablation, not a prospectively independent confirmation.
