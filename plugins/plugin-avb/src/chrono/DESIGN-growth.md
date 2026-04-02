# Chronometer chain growth & Merkle-commit redesign

This note quantifies on-disk growth of the v1 chain format under a
representative load of **one agent emitting 100 events per minute**, then
specifies the v2 format that replaces inline event payloads with a
**Merkle-root commitment** to an off-chain event log. v2 is the design we
are converging on.

---

## 1. Inputs

| Quantity                       | Value    | Source                                            |
| ------------------------------ | -------- | ------------------------------------------------- |
| Block-time target              | 60 s     | `AVB_CHRONO_BLOCK_MS` default                     |
| Blocks per day                 | 1 440    | derived                                           |
| Blocks per year                | 525 600  | derived                                           |
| Events per block               | 100      | assumption (1 agent · 100 ev/min)                 |
| Header size                    | 108 B    | `HEADER_SIZE` (`block.ts`)                        |
| Per-event fixed overhead       | 11 B     | `u8 type + i64 hostTs + u16 len`                  |
| Detail string length (UTF-8)   | see §1.1 | sampled from `AvbService.chronicle()` call sites  |

### 1.1 Detail-length distribution

Sampled from the strings the AVB pipeline actually records
(`service.ts → chronicle()`), with `runId` as a 36-char UUID:

| Event type        | Example detail                                   | Bytes |
| ----------------- | ------------------------------------------------ | ----: |
| `RUN_STARTED`     | `run=<uuid>`                                     |   ~40 |
| `PHASE_STARTED`   | `run=<uuid> phase=IMAGE_PHASE`                   |   ~55 |
| `PHASE_COMPLETE`  | `run=<uuid> phase=IMAGE_PHASE`                   |   ~55 |
| `PHASE_FAILED`    | `run=<uuid> phase=… err=<msg>`                   |   ~90 |
| `DELIVER`         | `run=<uuid> url=https://cdn.…/avatar.png`        |  ~120 |
| `BOOT`/`SHUTDOWN` | `agent=<uuid> height=N hostTs=N`                 |   ~70 |

We model three scenarios:

| Scenario | Mean detail `d̄` | Bytes / event = 11 + d̄ |
| -------- | --------------: | ----------------------: |
| **Low**  |            40 B |                    51 B |
| **Avg**  |            64 B |                    75 B |
| **High** |           120 B |                   131 B |

---

## 2. v1 (current): events inlined in the block

```
block = 108-byte header  ‖  Σ encoded events
```

Per-block size at 100 ev/block:

| Scenario | Payload          | Block total | Per minute | Per day  | Per month (30 d) | Per year |
| -------- | ---------------: | ----------: | ---------: | -------: | --------------: | -------: |
| Low      | 100·51  = 5 100 B |     5 208 B |    5.21 KB |  7.50 MB |          225 MB |  2.74 GB |
| **Avg**  | 100·75  = 7 500 B |     7 608 B |    7.61 KB | 10.96 MB |          329 MB |  4.00 GB |
| High     | 100·131 =13 100 B |    13 208 B |   13.21 KB | 19.02 MB |          571 MB |  6.94 GB |

Closed form:  `chainBytes(t_min) = t_min · (108 + 100 · (11 + d̄))`.

**Operational consequences of v1**

* `loadChain()` reads and SHA-256-verifies the **entire file** on every
  boot. At 4 GB/year that is ~10 s of hashing on a laptop just to start
  the agent, and grows linearly forever.
* The chain file is the integrity artifact **and** the event store, so it
  cannot be rotated, compressed, or pruned without breaking verification.
* Anyone wanting to prove "event *e* happened before block *N*" must ship
  the whole block (≈ 7.6 KB avg) — there is no per-event inclusion proof.

---

## 3. v2 (target): Merkle-root commitment, off-chain event log

### 3.1 Layout

```
┌──────────── chain.bin (tamper-evident, PoW-sealed) ───────────┐
│  block = 108-byte header only                                 │
│    eventsHash field  ← Merkle root of the block's events      │
│    eventBytes field  ← repurposed: byte length of the          │
│                         off-chain segment (for seek)           │
│  payload             ← 0 bytes                                  │
└────────────────────────────────────────────────────────────────┘

┌──────────── events.bin (off-chain, plain append log) ─────────┐
│  per block:  u32 height | u32 segLen | encoded events…         │
│  (same per-event encoding as v1)                                │
└────────────────────────────────────────────────────────────────┘
```

* **Leaf** *i* = `sha256( u8 0x00 ‖ encodedEvent_i )`
  (the `0x00` domain-sep prefix prevents second-preimage attacks where an
  interior node is passed off as a leaf — RFC 6962 style).
* **Interior** = `sha256( u8 0x01 ‖ left ‖ right )`; odd last leaf is
  duplicated (Bitcoin convention).
* **Root** is written into the existing 32-byte `eventsHash` header slot
  → **no header-size change**, `PROTOCOL_VERSION` bumps to `2`.
* `eventCount` stays as-is; `eventBytes` now records the off-chain
  segment length so a verifier can `seek()` directly without an index.

### 3.2 Chain growth

The chain file is now **constant 108 B / block**, independent of event
volume or detail length:

|                | Per minute | Per day   | Per month | Per year  |
| -------------- | ---------: | --------: | --------: | --------: |
| **chain.bin**  |      108 B |  155.5 KB |   4.67 MB |  56.76 MB |

vs v1-avg 4.00 GB/yr → **70× smaller chain** (51× at low, 123× at high).
Boot-time verification is `O(blocks)` hashing of 108-byte headers only:
~57 MB/yr ≈ 150 ms of SHA-256 to verify a year of history.

### 3.3 Off-chain log growth

`events.bin` carries what the v1 payload used to, plus an 8-byte frame
header per block:

| Scenario | Per minute        | Per day  | Per year |
| -------- | ----------------: | -------: | -------: |
| Low      | 8 + 5 100 = 5 108 B | 7.36 MB |  2.69 GB |
| **Avg**  | 8 + 7 500 = 7 508 B |10.81 MB |  3.95 GB |
| High     | 8 +13 100 =13 108 B |18.88 MB |  6.89 GB |

Total bytes on disk are essentially unchanged (v2 chain + log ≈ v1 chain),
**but** the log is now:

* **Cold** — never re-read on boot; only consulted when someone asks
  "what actually happened in block *N*?".
* **Compressible** — UTF-8 details with repeating `run=<uuid>` prefixes
  gzip to ~25–35 % (≈ 1–1.4 GB/yr avg).
* **Rotatable / archivable** — segments older than the operator's
  retention window can be shipped to object storage or deleted outright;
  the chain still proves *that* 100 events with Merkle root *R* existed
  at minute *M*, even if the bodies are gone.

### 3.4 Inclusion proofs

To prove a single event *e* was sealed into block *N*:

```
proof = { blockHeader (108 B),
          encodedEvent (11 + d̄ B),
          merklePath   (⌈log₂ eventCount⌉ × 32 B) }
```

At 100 events/block, `⌈log₂ 100⌉ = 7` → path = 224 B.
**Avg proof size ≈ 108 + 75 + 224 = 407 B**, vs ~7.6 KB to ship a whole
v1 block — ~19× smaller, and verifiable with 8 SHA-256 calls.

The verifier:

1. `sha256d(header)` ≤ `bitsToTarget(header.difficultyBits)`  → PoW valid.
2. Walk `merklePath` from `leaf(e)` → must equal `header.eventsHash`.
3. (Optionally) check `header.prevHash` links into a known-good tip.

---

## 4. Side-by-side

| Metric (100 ev/min, avg detail 64 B)     | v1 inline          | v2 Merkle-commit      | Δ       |
| ---------------------------------------- | -----------------: | --------------------: | ------- |
| `chain.bin` / minute                     |           7 608 B  |               108 B   | 70.4×   |
| `chain.bin` / year                       |           4.00 GB  |             56.8 MB   | 70.4×   |
| Bytes hashed on boot (1 yr)              |           4.00 GB  |             56.8 MB   | 70.4×   |
| Total disk / year (chain + log)          |           4.00 GB  | 56.8 MB + 3.95 GB     | ~1.0×   |
| …with gzip on log                        |              n/a   | 56.8 MB + ~1.2 GB     | ~3.3×   |
| Single-event proof size                  |          ~7 608 B  |              ~407 B   | 18.7×   |
| Event-body retention required for chain  |            forever |              optional | —       |
| Header format change                     |                 —  | version=2, field reuse | none structural |

---

## 5. Migration & implementation notes

* **No header reshape.** `eventsHash` (offset 44) becomes `merkleRoot`;
  `eventBytes` (offset 104) becomes `segmentBytes`. `HEADER_SIZE` stays
  108, so v1 and v2 blocks coexist in one file — `version` at offset 4
  discriminates. `loadChain()` skips payload read when `version ≥ 2`.
* **Miner unchanged.** The worker already hashes a fixed 108-byte header
  with `eventsHash` pre-filled; the service just computes a Merkle root
  instead of `sha256(payload)` before dispatching the job, and writes the
  drained events to `events.bin` instead of after the header.
* **New surface:**
  * `merkleRoot(events: ChronoEvent[]): Uint8Array`
  * `merkleProof(events, index): Uint8Array[]`
  * `verifyInclusion(header, event, path): boolean`
  * `readSegment(eventsPath, height): ChronoEvent[]` (lazy, for tooling)
* **Integrity codes** gain `MERKLE_ROOT_MISMATCH` (off-chain segment
  re-hashed ≠ header root) — only checked on demand, not on boot.
* **Genesis** stays v1 (single `GENESIS` event, root = leaf hash) so an
  empty `events.bin` is valid.

This is the design targeted for the next change set.
