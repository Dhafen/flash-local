# Prepared-cache upgrade recovery

Implemented on `codex/ornith-prompt-controls`; replacement build and physical
upgrade acceptance are pending. Build 126 is distributed but can reject files
prepared by the preceding engine revision. Keep the app and installed GGUF.

The reported `LES prepared files failed integrity verification` occurs before
native inference: the old prepared seal records the previous runtime revision.
The directory name depends only on the model SHA, so build 126 finds that old
directory and throws instead of regenerating it. Fresh-directory host tests did
not exercise this app-upgrade path.

The correction treats prepared dense weights as a reproducible cache. Reuse
still requires the verified source SHA/length, current runtime revision and both
payload hashes. Old, missing, malformed or damaged cache files regenerate from
the already verified local GGUF descriptor. A failed source identity check stops
the operation before changing the prepared files. Integrity and numerical
admission are not bypassed or replaced with a new signature on old bytes.

One staging directory and one recoverable previous directory are reserved for
each model hash under private application storage. The old directory remains
until the replacement is fully written, read back, hashed and sealed. Publication
uses recoverable renames with rollback; startup can finish an interrupted valid
transaction. Cancelled/failed preparation preserves the existing directory.
Cleanup is restricted to the exact private derived-cache paths, rejects directory
symlinks and never operates on the original model or conversation/diagnostic
storage. No live native context may own the files during preparation.

The copy is only the fixed-weight portion (approximately 1.4 GB for this model),
not another 21.7 GB GGUF. Available capacity is checked before reconstruction;
the same bounded native copy/readback buffers and RAM admission are retained.
The app displays that it is rebuilding prepared files. No model download,
requantization, routing, sampler or context-limit change is involved.

Five new generated-fixture regression functions cover:

- A valid old-revision seal: actual reconstruction, exact dense bytes and exact
  two-forward CPU-control logits before/after, without changing the original.
- Damaged dense bytes, missing manifest/seal, malformed seal and source-length
  mismatch in the derived seal: repair from the verified source.
- Interrupted publication, complete staged-file recovery, valid-cache reuse and
  cleanup of obsolete private transaction files.
- Cancellation retaining the old cache, and directory-symlink rejection without
  touching unrelated files.
- An altered original GGUF: rejection before changing the prepared cache.

These are host controls with tiny untrained fixtures. The actual phone still
needs to confirm startup over its existing cache, fresh numerical admission and
the batch benchmark. No larger-batch speed result is established by this repair.
The accompanying UI change gives both batch menus explicit visible labels on
iPhone and Mac; new captures must be reviewed before delivery is confirmed.
