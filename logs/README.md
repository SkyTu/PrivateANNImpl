# Saved measurements

Small CSV, JSON, log and process-time reports are committed as evidence of the
documented experiments. Reports can describe older implementations or failed
diagnostics; consult `TEST_STATUS.md` for their scope and interpretation.

Large Nsight/SQLite files and per-call `*_runtime.csv` / `*_all_kernels.csv`
traces stay local. Their links in historical reports refer to artifacts produced
by the documented profiling commands and are not present in a fresh clone.
No key or ciphertext dumps are included.

The one-million-record component report is
`experiment/tfhe_rs/REPORT_1M_C500_U15_B100_K10.md`; its source measurements are
`cluster_bucket_once.csv`, `cluster_bucket_once.log` and `cluster_bucket_once.time`.
