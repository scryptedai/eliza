/**
 * Chronometer barrel — public surface of the on-device PoW timestamp server.
 */

export {
  bitsToTarget,
  decodeEvents,
  encodeEvent,
  encodeEvents,
  HEADER_SIZE,
  INITIAL_DIFFICULTY_BITS,
  meetsTarget,
  PROTOCOL_VERSION,
  retarget,
  splitEvents,
  targetToBits,
  toHex,
} from "./block.ts";
export {
  CHRONO_BRIDGE_FORWARDED,
  CHRONO_BRIDGE_SKIPPED,
  chronoRuntimeEvents,
} from "./bridge.ts";
export {
  loadChain,
  loadEventSegments,
  parseChain,
  verifyBlocks,
  verifySegments,
} from "./chain.ts";
export {
  leafHash,
  type MerkleStep,
  merkleProof,
  merkleRoot,
  rootFromProof,
} from "./merkle.ts";
export {
  CHRONO_BLOCK_SEALED_EVENT,
  CHRONO_INTEGRITY_EVENT,
  CHRONO_SERVICE_TYPE,
  ChronometerService,
  ENV_CHRONO_BLOCK_MS,
  ENV_CHRONO_CPU_PERCENT,
  ENV_CHRONO_DATA_DIR,
  ENV_CHRONO_ENABLED,
  type InclusionProof,
  verifyInclusion,
} from "./service.ts";

export {
  type Block,
  type BlockHeader,
  type ChronoEvent,
  ChronoEventType,
  type IntegrityCode,
  type IntegrityIssue,
} from "./types.ts";
