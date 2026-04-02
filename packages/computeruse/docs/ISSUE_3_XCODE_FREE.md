# Issue #3 — Xcode-free build path analysis

**Companion to:** [UPGRADE_PLAN.md](./UPGRADE_PLAN.md) §Issue #3
**Status:** ⚠️ **WIP / Incomplete** — research verified, recommended fix not yet validated end-to-end
**Date:** 2026-04-02

---

## TL;DR

**Issue #3 itself does not require Xcode.** The dependency that needs `xcodebuild` is `cidre`, which is OCR-only and entirely orthogonal to the AX un-stubbing work. The two problems were conflated because they fail in the same `cargo build` invocation.

| Concern | Crate | Needs Xcode? | Blocks #3? |
|---|---|---|---|
| AX tree traversal (the actual #3 work) | `accessibility`, `accessibility-sys` | **No** — `build = false` | — |
| OCR on macOS | `cidre` ← `uni-ocr` | Yes (cidre 0.9.x only) | No, orthogonal |

**Recommended fix:** `[patch.crates-io]` cidre to 0.15.0, which rewrote its build to use the `cc` crate instead of `xcodebuild`. Builds with Command Line Tools alone.

---

## Why this looked like an Xcode problem

The current dev-loop workaround (`DOCS_RS=1` + stub `.a` archives) was developed because `cargo build -p computeruse-rs` fails on a CLT-only machine with:

```
error: failed to run custom build command for `cidre v0.9.2`
  ...
  xcode-select: error: tool 'xcodebuild' requires Xcode, but active developer
  directory '/Library/Developer/CommandLineTools' is a command line tools instance
```

Because `computeruse-rs` is the crate where `macos/mod.rs` lives, the assumption was that working on #3 requires either Xcode or an elaborate stub. **It doesn't.** The build failure comes from a sibling subtree of the dependency graph that #3 never touches.

---

## Evidence: the AX crates have no build script

```toml
# ~/.cargo/registry/.../accessibility-0.2.0/Cargo.toml:17
build = false

# ~/.cargo/registry/.../accessibility-sys-0.2.0/Cargo.toml:17
build = false
```

These are pure `extern "C"` declarations against `ApplicationServices.framework`, which ships with macOS itself. No codegen, no ObjC bridge, no `.m` files. They link with `-framework ApplicationServices` and that's it.

The infrastructure is already proven working in this codebase:

```rust
// crates/computeruse/src/platforms/macos/mod.rs:5-9
use accessibility::{AXAttribute, AXUIElement, AXUIElementAttributes, Error as AxError};
use accessibility_sys::{
    kAXPositionAttribute, kAXSizeAttribute, AXUIElementCopyAttributeValue, AXValueGetType,
    AXValueGetValue, AXValueRef, kAXValueTypeCGPoint, kAXValueTypeCGSize,
};
```

…and called successfully at [`mod.rs:202`](../crates/computeruse/src/platforms/macos/mod.rs) and `:211` for bounds queries. The 76 stubs are unimplemented method *bodies*, not missing FFI surface. Filling them in is `cargo build`-clean today on this machine — *if* the unrelated cidre build is bypassed.

---

## Where Xcode actually enters the graph

```
computeruse-rs
├── accessibility ──────────────── build = false ✓
├── accessibility-sys ─────────── build = false ✓
└── uni-ocr 0.1.5
    └── cidre 0.9.2 ──────────── build.rs invokes xcodebuild ✗
        (target.'cfg(target_os = "macos")'.dependencies, non-optional)
```

[`cidre-0.9.2/build.rs:121`](https://docs.rs/crate/cidre/0.9.2/source/build.rs):

```rust
} else {
    Command::new("xcodebuild")
        .args(["-project", "./pomace/pomace.xcodeproj"])
        .args(["-sdk", sdk])
        // ...
```

This is the unconditional `aarch64-apple-darwin` path. `pomace` is a 26-target Xcode project that compiles ObjC bridge stubs for `NSArray`, `VNRecognizeTextRequest`, etc. — the symbols our current `libns.a`/`libvn.a` workaround fakes.

The `DOCS_RS=1` escape hatch lives at [`build.rs:233`](https://docs.rs/crate/cidre/0.9.2/source/build.rs):

```rust
fn main() {
    if std::env::var("DOCS_RS").is_ok() {
        return;
    }
```

…which is why the current workaround functions, at the cost of any test that touches OCR segfaulting on a null `NS_ARRAY` deref.

---

## The three workarounds

### Option 1 — Patch cidre to 0.15 ⭐ **Recommended**

cidre 0.15.0 (current latest) **rewrote `build.rs` to drop xcodebuild entirely** for native macOS targets. Instead of compiling `pomace.xcodeproj`, it compiles individual `.m` source files via the `cc` crate:

```rust
// cidre-0.15.0/build.rs:63
let mut build = cc::Build::new();
build.file(&src);
build.flag("-fobjc-arc");
// ...
build.compile(name);
```

`cc` invokes `clang` directly. The only remaining `Command::new` calls in 0.15:

| Line | Command | When it fires |
|---|---|---|
| [`build.rs:271`](https://docs.rs/crate/cidre/0.15.0/source/build.rs) | `clang --print-search-dirs` | Always — but `clang` ships with CLT |
| [`build.rs:331`](https://docs.rs/crate/cidre/0.15.0/source/build.rs) | `xcrun --show-sdk-path` | Only `sdk == "maccatalyst"` AND `SDKROOT` unset |

For `aarch64-apple-darwin` the `xcrun` branch is dead. Verified on this machine:

```console
$ which clang && clang --version | head -1
/usr/bin/clang
Apple clang version 17.0.0 (clang-1700.0.13.5)
$ ls /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk
[exists]
```

Everything `cc::Build` needs is present without Xcode.

**Implementation** — workspace root `Cargo.toml`:

```toml
[patch.crates-io]
# cidre 0.9.x build.rs invokes xcodebuild; 0.15 uses cc::Build (clang only).
# uni-ocr pins ^0.9 so we override at the workspace level.
cidre = "=0.15.0"
```

**Risk:** uni-ocr's `apple.rs` imports `cidre::{cv::{PixelBuf, PixelFormat}, ns, vn::{ImageRequestHandler, RecognizeTextRequest}}`. These types still exist in 0.15 (`src/vn/request_handler.rs`, `src/vn/recognize_text_request.rs`) but **6 minor versions of API drift have not been compile-tested against uni-ocr's call sites yet.** If signatures changed, the patch becomes:

```toml
[patch.crates-io]
cidre = "=0.15.0"
uni-ocr = { git = "https://github.com/elizaOS/uni-ocr", branch = "cidre-0.15" }
```

…with a ~50-LOC fork adjusting `apple.rs:77` onward. Still no Xcode.

**Why preferred:** It's the only option that makes OCR actually *work* on CLT-only machines, rather than stubbing it out. Removes the entire `DOCS_RS` + stub-archive contraption from the dev loop. One workspace-level patch, zero source changes if the API held.

---

### Option 2 — Feature-gate the macOS OCR call sites

`uni-ocr` itself has no `[features]` table and `cidre` is not `optional = true`, so this can't be done at the uni-ocr level. But `computeruse-rs` only calls `uni_ocr::OcrEngine` from three places on macOS:

| File | Line | Context |
|---|---|---|
| [`element.rs`](../crates/computeruse/src/element.rs) | 1520 | `UIElement::ocr()` — cross-platform method body |
| [`macos/mod.rs`](../crates/computeruse/src/platforms/macos/mod.rs) | 1936 | `MacOSEngine::ocr_screenshot` |
| [`macos/mod.rs`](../crates/computeruse/src/platforms/macos/mod.rs) | 1960 | `MacOSEngine::ocr_image_path` |

(Windows and Linux have their own call sites, but those don't pull in cidre.)

Add to `computeruse-rs/Cargo.toml`:

```toml
[features]
default = ["ocr"]
ocr = ["dep:uni-ocr"]

[target.'cfg(target_os = "macos")'.dependencies]
uni-ocr = { workspace = true, optional = true }
```

Gate the three call sites with `#[cfg(feature = "ocr")]` and provide `#[cfg(not(feature = "ocr"))]` stubs returning `UnsupportedOperation("built without ocr feature")`.

Dev loop becomes:

```bash
cargo build -p computeruse-rs --no-default-features
```

**Pro:** Zero upstream churn. Removes `DOCS_RS=1` and the stub archives. **Con:** OCR is silently absent in dev builds — the divergence between dev and release behavior is exactly the kind of thing that bites later. Doesn't fix CI.

---

### Option 3 — Keep the stub workaround as-is

Already works for the non-OCR test surface (49/49 across `browser_backend` + `ref_snapshot` + `close_tab`):

```bash
DOCS_RS=1 \
RUSTFLAGS="-L $PWD/packages/computeruse/target/cidre-stub-libs -l static=ns -l static=vn" \
  cargo test -p computeruse-rs --lib
```

Document it in `CONTRIBUTING.md`, do nothing else.

**Pro:** Zero changes. **Con:** Fragile (`cargo clean` deletes the stubs), opaque to new contributors, OCR tests segfault with no useful message, and the `RUSTFLAGS` global poisons `rust-analyzer` if you set it in `.cargo/config.toml`. Pure debt.

---

## What "fixing #3" actually requires

With cidre out of the way (any of the three options), the remaining constraints on #3 are **runtime, not build-time**:

| Constraint | Severity | Mitigation |
|---|---|---|
| Accessibility permission grant | Hard — AX calls return `kAXErrorAPIDisabled` without it | `AXIsProcessTrusted()` preflight (already in `accessibility-sys`); see UPGRADE_PLAN §Phase 4 |
| Real window to query | Hard — can't unit-test `children()` against thin air | Spawn `TextEdit` in the test, query its window. Mark `#[ignore]` and run manually / in macOS CI |
| CI runner permission | Soft — GitHub macOS runners don't grant AX by default | `sudo sqlite3 /Library/Application\ Support/com.apple.TCC/TCC.db ...` in workflow setup, or `tccutil` |

None of these involve a compiler.

---

## Recommendation

1. **Now:** Land Option 1 (`[patch.crates-io] cidre = "0.15"`). If the uni-ocr API drift is non-trivial, fall back to a thin uni-ocr fork on the same patch entry.
2. **Then:** Delete `target/cidre-stub-libs/` and the `DOCS_RS=1` instructions.
3. **Then:** Implement #3 phases per UPGRADE_PLAN — Phase 1 (`children`/`parent`) is ~80 LOC and unblocks selectors immediately.

---

## Open items (why this doc is WIP)

- [ ] Compile-test `[patch.crates-io] cidre = "0.15"` against `uni-ocr-0.1.5/src/apple.rs` — confirm whether a uni-ocr fork is needed
- [ ] Verify `cc::Build` finds the macOS SDK without `SDKROOT` set on a fresh CLT-only machine (it should via `clang`'s default search paths, but unconfirmed)
- [ ] Confirm `cidre-0.15`'s `vn` feature flag set matches what `uni-ocr` requests (`cf, cg, cv, cm, ns, objc, vn, blocks, macos_13_0`)
- [ ] Decide whether the uni-ocr fork (if needed) lives under elizaOS org or as a workspace member
