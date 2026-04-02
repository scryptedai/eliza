# AVB Chronometer

On-device proof-of-work timestamp server. Gives an Autonomous Virtual Being a
**tamper-evident intrinsic clock** that is independent of the host system
clock (which an admin can edit) and that records what the agent did, when.

The Chronometer continuously mines a private hash-linked chain at a fixed
share of total machine CPU. Each block seals a binary log of the AVB events
that occurred since the previous block, plus the host's *reported* time. On
restart the chain is reloaded and verified; any inconsistency — broken hash
linkage, missing PoW, host-clock regression, suspicious gap — raises an
integrity warning that is itself sealed into the next block.

```
┌──────────────┐  recordEvent()           ┌───────────────┐
│  AvbService  │ ───────────────────────▶ │ Chronometer   │
│  (pipeline)  │                          │   Service     │
└──────────────┘                          └──────┬────────┘
                                                 │ MineJob {…, eventsHash = merkleRoot(events)}
                                                 ▼
                                         ┌────────────────┐
                                         │  miner-worker  │  duty-cycled
                                         │ (worker_thread)│  sha256d PoW
                                         └──────┬─────────┘
                                                │ sealed 108-B header
                                                ▼
              <dataDir>/logs/events.chain  ◀─── append header (no payload)
              <dataDir>/logs/events.bin    ◀─── append segment {height,len,events}
```

---

## Storage layout (v2)

Per agent, under `AVB_CHRONO_DATA_DIR` (default
`${ELIZA_DATA_DIR}/avb-chrono/<agentId>`):

```
logs/
  events.chain   header-only PoW chain — exactly 108 B/block, ~57 MB/yr
  events.bin     off-chain event segments, Merkle-committed by the chain
```

`events.chain` is the trust anchor: each header carries the **Merkle root**
of that block's events, so any byte changed in `events.bin` is detectable by
recomputing the root. `events.bin` can be compressed, rotated, or pruned
independently — a verifier holding only the chain plus a single encoded
event and its O(log n) Merkle path can still confirm the event was sealed
(see *Inclusion proofs* below).

`events.bin` framing, one segment per block:
`u32 height (LE) | u32 segLen (LE) | segLen bytes of encoded events`.

> **v1 back-compat** — version-1 blocks inline `eventBytes` of payload after
> the header in the chain file and set `eventsHash = sha256(payload)`. The
> reader walks both versions transparently; only `EVENTS_HASH_MISMATCH`
> applies to v1 and only `MERKLE_ROOT_MISMATCH` to v2.

---

## Configuration

| Env / setting              | Default  | Meaning                                                                                                   |
| -------------------------- | -------- | --------------------------------------------------------------------------------------------------------- |
| `AVB_CHRONO_CPU_PERCENT`   | `5`      | Percent of **total machine CPU** to allocate. Translated to a duty cycle on one core: `cores × pct%`, capped at 100 % (single miner thread). |
| `AVB_CHRONO_BLOCK_MS`      | `60000`  | Block-time target. Difficulty retargets toward this every block (¼×–4× clamp).                           |
| `AVB_CHRONO_DATA_DIR`      | `${ELIZA_DATA_DIR}/avb-chrono/<agentId>` | Per-agent directory containing `logs/events.chain` and `logs/events.bin`. |
| `AVB_CHRONO_ENABLED`       | `true`   | Set `false`/`0` to disable the miner thread (tests, CI). Genesis is still minted; events still buffered. |

---

## Service API

`ChronometerService` is registered on the AVB plugin and started before
`AvbService` so the pipeline can record into it from boot.

```ts
import {
  CHRONO_SERVICE_TYPE,
  ChronoEventType,
  type ChronometerService,
} from "@elizaos/plugin-avb";

const chrono = await runtime.getServiceLoadPromise(
  CHRONO_SERVICE_TYPE,
) as ChronometerService;

chrono.recordEvent(ChronoEventType.OTHER, "something happened");
chrono.getHeight();           // tip block height (0 = genesis)
chrono.getIntrinsicTimeMs();  // Σ wallElapsedMs across all blocks — the
                              // agent's personally-witnessed PoW clock
chrono.tip();                 // Block at the tip
chrono.getIntegrityIssues();  // readonly IntegrityIssue[] seen so far
chrono.verify();              // re-check in-memory chain headers, flag issues
await chrono.verifyEventLog();// cross-check events.bin against sealed Merkle
                              // roots (heavier; on-demand only)
chrono.getDataDir();          // per-agent directory
chrono.getChainPath();        // …/logs/events.chain
chrono.getEventsPath();       // …/logs/events.bin

// Inclusion proof: prove one event was sealed without shipping events.bin.
const proof = await chrono.proveInclusion(/*height*/ 42, /*eventIndex*/ 3);
//   proof = { header: BlockHeader, encodedEvent: Uint8Array, path: MerkleStep[] }
verifyInclusion(proof); // → true; checks header PoW + Merkle path → eventsHash
```

`recordEvent()` is non-blocking; the event is buffered and drained into the
**next** dispatched mining job, so it appears in the block after the one
currently being mined.

### Runtime events emitted

| Event name                       | Payload                                                                                |
| -------------------------------- | -------------------------------------------------------------------------------------- |
| `AVB_CHRONO_BLOCK_SEALED`        | `{ height, hash, hostTimestampMs, wallElapsedMs, eventCount }` — every sealed block.   |
| `AVB_CHRONO_INTEGRITY_WARNING`   | `IntegrityIssue` — on load or `verify()` when tampering / clock anomalies are detected. |

---

## Event types

`ChronoEventType` (`u8` on the wire):

| Code | Name               | Recorded by                                            |
| ---- | ------------------ | ------------------------------------------------------ |
| 0    | `GENESIS`          | Service, once, when minting block 0.                  |
| 1    | `BOOT`             | Service, every `start()`.                              |
| 2    | `RUN_STARTED`      | `AvbService.createRun()`                               |
| 3    | `PHASE_STARTED`    | `AvbService.spawnPhaseTask()`                          |
| 4    | `PHASE_COMPLETE`   | `AvbService.onPhaseComplete()`                         |
| 5    | `PHASE_FAILED`     | `AvbService.onPhaseFailed()`                           |
| 6    | `DELIVER`          | `AvbService.deliverSuccess()`                          |
| 7    | `INTEGRITY_WARNING`| Service, whenever an issue is flagged (so suspicion is itself sealed). |
| 8    | `SHUTDOWN`         | Service, on `stop()`.                                  |
| 9    | `RUNTIME`          | `chrono/bridge.ts` — forwarded from a core `EventType`. |
| 255  | `OTHER`            | Free-form.                                             |

### Automatic capture of ElizaOS runtime events

The AVB plugin registers `chronoRuntimeEvents` on its `events` field so
that **notable** `@elizaos/core` `EventType` emissions are mirrored into
the chronometer without any explicit `recordEvent()` call:

* **Forwarded** (`CHRONO_BRIDGE_FORWARDED`): messaging
  (`MESSAGE_RECEIVED/SENT`, `REACTION_RECEIVED`, `INTERACTION_RECEIVED`,
  `POST_GENERATED`), topology (`WORLD_*`, `ENTITY_JOINED/LEFT`),
  reasoning loop (`RUN_*`, `ACTION_*`, `EVALUATOR_COMPLETED`,
  `MODEL_USED`), and process lifecycle (`HOOK_SESSION_*`,
  `HOOK_AGENT_START/END`, `HOOK_GATEWAY_*`, `HOOK_COMMAND_*`,
  `HOOK_COMPACTION_*`).
* **Skipped as spam** (`CHRONO_BRIDGE_SKIPPED`): `VOICE_*`,
  `EMBEDDING_*`, `HOOK_TOOL_*`, `HOOK_MESSAGE_SENDING`,
  `CONTROL_MESSAGE`, `MESSAGE_DELETED`, `CHANNEL_CLEARED`,
  `ENTITY_UPDATED`, `EVALUATOR_STARTED`, `FORM_*`,
  `HOOK_AGENT_BOOTSTRAP`, `ROOM_*`.

Each forwarded event becomes a single `RUNTIME` chrono event whose
detail string is `"<EventType> <k=v …>"` — IDs truncated to 8 chars,
free text clipped to 64 chars, total capped at 240 bytes. A unit test
asserts that *every* `EventType` value is explicitly listed in one set
or the other, so adding a new core event fails CI until someone decides
whether it is notable.

---

## Integrity codes

Returned by `loadChain()` / `verifyBlocks()` and surfaced via
`AVB_CHRONO_INTEGRITY_WARNING`:

| Code                    | Meaning                                                                                                            |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `BAD_MAGIC`             | A header doesn't begin with `"AVBC"` — file is corrupt or not a chain file.                                       |
| `TRUNCATED`             | File ends mid-block (partial write / crash). Trailing bytes are ignored; chain resumes from the last whole block. |
| `HEIGHT_MISMATCH`       | Block height ≠ previous + 1.                                                                                       |
| `PREV_HASH_MISMATCH`    | `prevHash` ≠ hash of previous block — chain was spliced or a block was replaced.                                  |
| `EVENTS_HASH_MISMATCH`  | (v1 only) `sha256(payload)` ≠ `eventsHash` — inline payload tampered after sealing.                              |
| `MERKLE_ROOT_MISMATCH`  | (v2) Off-chain segment for a block is missing, wrong length, or its recomputed Merkle root ≠ `eventsHash`. Raised by `verifyEventLog()` / `verifySegments()`. |
| `POW_INVALID`           | `sha256d(header)` > target(difficultyBits) — header was edited without re-mining.                                 |
| `HOST_CLOCK_REGRESSION` | Host timestamp went backwards between consecutive blocks.                                                         |
| `HOST_CLOCK_GAP`        | Host-time delta ≫ PoW-measured `wallElapsedMs` (heuristic: > 5 min **and** > 8× wall) — possible suspend/restore.  |

---

## Block format (on-disk, little-endian)

108-byte header. **v2** writes nothing after the header in `events.chain`;
**v1** follows it with `eventBytes` of inline payload.

| Offset | Size | Field            | Notes                                                                  |
| -----: | ---: | ---------------- | ---------------------------------------------------------------------- |
|      0 |    4 | magic            | ASCII `AVBC`                                                           |
|      4 |    4 | version          | u32, currently `2`                                                     |
|      8 |    4 | height           | u32                                                                    |
|     12 |   32 | prevHash         | sha256d of previous header                                            |
|     44 |   32 | eventsHash       | v2: RFC-6962 Merkle root over per-event leaves · v1: sha256(payload)  |
|     76 |    8 | hostTimestampMs  | i64, `Date.now()` at job dispatch                                      |
|     84 |    4 | wallElapsedMs    | u32, miner's monotonic measurement                                     |
|     88 |    4 | difficultyBits   | u32, compact nBits target                                              |
|     92 |    8 | nonce            | u64                                                                    |
|    100 |    4 | eventCount       | u32                                                                    |
|    104 |    4 | eventBytes       | u32 — v2: length of the off-chain segment · v1: inline payload length |

Block hash = `sha256d(header[0..108])`. PoW: hash interpreted as a big-endian
256-bit unsigned integer must be ≤ `bitsToTarget(difficultyBits)`.

Per-event encoding (in `events.bin` segments / v1 inline payload):
`u8 type | i64 hostTs | u16 detailLen | detailLen bytes UTF-8`.

Merkle tree (v2): leaf = `sha256(0x00 ‖ encodedEvent)`,
interior = `sha256(0x01 ‖ L ‖ R)`, last odd leaf duplicated,
empty block → all-zero root.

---

## Mining mechanics

* **Thread** — a single `node:worker_threads` Worker (`miner-worker.ts`),
  `unref()`'d so it never keeps the process alive.
* **Duty cycle** — alternates `WORK_SLICE_MS = 50 ms` of hashing with
  `50 · (1/f − 1) ms` of sleep, where `f = min(1, cores · pct/100)`. E.g.
  5 % total on a 10-core box → `f = 0.5` → 50 ms on / 50 ms off.
* **Retarget** — after each sealed block,
  `nextTarget = prevTarget · wallElapsedMs / AVB_CHRONO_BLOCK_MS`,
  ratio clamped to [¼, 4], capped at the genesis ceiling. The first few
  blocks after genesis are near-instant while difficulty ramps up 4× per
  step from the trivial floor; convergence to the 60 s target typically
  takes ~8 blocks.
* **wallElapsedMs** is part of the hashed header. The worker refreshes it at
  the start of each 50 ms slice so the value sealed into a winning block is
  accurate to within one slice without a post-hoc re-hash.

---

## Offline tooling

```ts
import {
  loadChain, loadEventSegments, verifySegments,
  merkleRoot, splitEvents, decodeEvents, toHex, verifyInclusion,
} from "@elizaos/plugin-avb";

const dir = "/path/to/agent/logs";
const { blocks, issues } = await loadChain(`${dir}/events.chain`);
const segments = await loadEventSegments(`${dir}/events.bin`);

// Cross-check the off-chain log against sealed Merkle roots.
const segIssues = verifySegments(blocks, segments,
  (seg) => merkleRoot(splitEvents(seg)));
console.log("chain issues:", issues, "segment issues:", segIssues);

for (const b of blocks) {
  console.log(b.header.height, toHex(b.hash));
  for (const e of decodeEvents(segments.get(b.header.height) ?? new Uint8Array()))
    console.log("  ", e);
}
```

Or run the live demo from repo root:

```sh
bun plugins/plugin-avb/scripts/prove-chrono.ts
# overrides: CHRONO_PROVE_CPU, CHRONO_PROVE_BLOCK_MS, CHRONO_PROVE_BLOCKS
```
