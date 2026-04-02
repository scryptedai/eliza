# Vendored dependencies

## `cidre/` (v0.9.2, patched)

**Why vendored:** `uni-ocr 0.1.5` (transitive dep of `computeruse-rs` on macOS)
pins `cidre = "^0.9.0"`. Upstream `cidre 0.9.2`'s `build.rs` shells out to
`xcodebuild -project ./pomace/pomace.xcodeproj`, which fails on hosts that
only have Xcode Command Line Tools (no full Xcode.app).

**What changed vs crates.io `cidre 0.9.2`:**
1. `build.rs` — replaced verbatim with the `cc`-crate-based build script from
   `cidre 0.15.0`, which compiles `pomace/{name}/{name}.m` directly via clang
   (works with CLT). No source-level API changes.
2. `Cargo.toml` — appended `[build-dependencies] cc = "1"`.

Everything under `src/` and `pomace/` is byte-identical to the published
`cidre 0.9.2` crate.

**Activated via** `[patch.crates-io] cidre = { path = "vendor/cidre" }` in the
workspace `Cargo.toml`.

**Upstream removal path:** drop this directory and the `[patch.crates-io]`
entry once `uni-ocr` upgrades to `cidre >= 0.15` (which natively uses `cc`).
