# E9 verification

Workspace: `D:\Code\Python\M3SR-MambaIRv2`.

- Verified A/B/C branches, +0.02/+0.05 boundaries, failure precedence, exact weak-band/approximate-failure gap, uncorrected mechanisms and REVIEW handling.
- Complete synthetic five-dataset E5 eight-condition and E7 paired-image CSV fixtures exercise full 5/14/100/100/109 image counts, matched seeds, recomputed means/std, E7 10,000-resample NumPy bootstrap and report/CSV reconciliation.
- End-to-end JSON/Markdown/CSV outputs are deterministic. Missing evidence stays insufficient; altered CI, changed checkpoint file and duplicate image keys are rejected. E8 is optional.
- Synthetic fixtures and fake checkpoint bytes exist only in temporary test directories, clearly marked synthetic; none are actual SR results or published measured evidence.
- E9 imports no torch/BasicSR/CUDA code and changes no prior architecture, loss, dataset path, training configuration or E8 implementation.
- CLI help was verified with Python `-S` (no site packages); Python 3.8 AST syntax and Git whitespace checks passed. Running against the real local workspace correctly produced `insufficient_evidence / UNDETERMINED` and exit code 2, without an A/B/C decision.

Actual E4–E7 reports, source weights and per-image CSVs are not present in this workspace. No real A/B/C decision, new model training, real-data evaluation or new benchmark score is claimed. Existing E8 production-extension limitations remain documented in `E8_VERIFICATION.md`; they do not block E9 report processing once real reports/CSVs are supplied.
