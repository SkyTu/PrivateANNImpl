# Dependency patches

These patches are generated against the exact upstream commits in
`../versions.json`. Apply them with `python3 third_party/apply_patches.py` after
fetching the dependency sources; `--check` validates without modifying files.

- `heongpu.patch`: locate NTL at configure time and export the required linkage.
- `rngongpu.patch`: use ordinary CUDA allocations on devices without memory pools.
- `tfhe-rs.patch`: configurable CUDA architecture, memory-pool capability checks,
  and the optional contiguous fallback arena. The new arena header is included.

The arena remains experimental and disabled by default. Its synchronous release
path still introduces device-wide waits; the measured results and limitations
are documented in `../../TEST_STATUS.md`. These patches do not implement batched
comparison kernels or remove the upstream stream synchronizations.

Upstream copyright notices and licenses remain in the downloaded repositories;
applying these patches does not remove or replace them.
