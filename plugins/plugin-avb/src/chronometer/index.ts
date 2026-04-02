/**
 * Chronometer barrel exports.
 *
 * Public surface kept intentionally narrow: callers interact with the
 * service (recordEvent, getStatus) and may want the EventKind enum and
 * status types. Everything else is implementation detail.
 */

// Lower-level pieces — exported for testing and for any future
// out-of-process tooling (chain inspector CLI, etc).
export { loadChain, validateBlock } from "./chain.ts";
export {
  GENESIS_BITS,
  retarget,
  TARGET_BLOCK_MS,
} from "./difficulty.ts";
export {
  bitsToTarget,
  computeEventsRoot,
  headerHash,
  sha256d,
} from "./encoding.ts";
export { mineSync, verifyPow } from "./miner.ts";
export {
  CHRONOMETER_SERVICE_TYPE,
  ChronometerService,
  type ChronometerStatus,
  ENV_AVB_CHRONOMETER_ENABLED,
  ENV_AVB_CHRONOMETER_PATH,
} from "./service.ts";
export {
  type ChainIssue,
  type ChainIssueCode,
  type ChainTip,
  type ChronoEvent,
  EventKind,
  type ValidationReport,
} from "./types.ts";
