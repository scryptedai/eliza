/**
 * Merkle commitments over per-event leaves.
 *
 * RFC-6962-style domain separation:
 *   leaf     = SHA256( 0x00 ‖ encodedEvent )
 *   interior = SHA256( 0x01 ‖ left ‖ right )
 * with the Bitcoin convention that an odd trailing node is paired
 * with itself.
 *
 * The 0x00/0x01 prefix prevents a second-preimage attack where an
 * interior hash is presented as a leaf.
 *
 * Empty tree → 32 zero bytes (matches ZERO_HASH so an empty block's
 * eventsHash is recognisably "no events").
 */

import { HASH_SIZE, sha256, ZERO_HASH } from "./block.ts";

const LEAF_PREFIX = 0x00;
const NODE_PREFIX = 0x01;

export function leafHash(encodedEvent: Uint8Array): Uint8Array {
  const buf = new Uint8Array(1 + encodedEvent.length);
  buf[0] = LEAF_PREFIX;
  buf.set(encodedEvent, 1);
  return sha256(buf);
}

function nodeHash(left: Uint8Array, right: Uint8Array): Uint8Array {
  const buf = new Uint8Array(1 + HASH_SIZE * 2);
  buf[0] = NODE_PREFIX;
  buf.set(left, 1);
  buf.set(right, 1 + HASH_SIZE);
  return sha256(buf);
}

/** Build all levels bottom-up. levels[0] = leaves, levels[last] = [root]. */
function buildLevels(leaves: Uint8Array[]): Uint8Array[][] {
  if (leaves.length === 0) return [[ZERO_HASH]];
  const levels: Uint8Array[][] = [leaves];
  let cur = leaves;
  while (cur.length > 1) {
    const next: Uint8Array[] = [];
    for (let i = 0; i < cur.length; i += 2) {
      const l = cur[i];
      const r = i + 1 < cur.length ? cur[i + 1] : cur[i];
      next.push(nodeHash(l, r));
    }
    levels.push(next);
    cur = next;
  }
  return levels;
}

/** Merkle root over `encodedEvents` (each a single encoded ChronoEvent). */
export function merkleRoot(encodedEvents: readonly Uint8Array[]): Uint8Array {
  const levels = buildLevels(encodedEvents.map(leafHash));
  return levels[levels.length - 1][0];
}

/**
 * One step of an inclusion path.
 * `right` = true means the sibling sits on the RIGHT (so combine as
 * nodeHash(running, sibling)); false means sibling is on the left.
 */
export interface MerkleStep {
  sibling: Uint8Array;
  right: boolean;
}

/**
 * Inclusion proof for `encodedEvents[index]`.
 * Returns the sibling path from leaf to root (root excluded).
 */
export function merkleProof(
  encodedEvents: readonly Uint8Array[],
  index: number,
): MerkleStep[] {
  if (index < 0 || index >= encodedEvents.length) {
    throw new RangeError(`merkleProof: index ${index} out of range`);
  }
  const levels = buildLevels(encodedEvents.map(leafHash));
  const path: MerkleStep[] = [];
  let idx = index;
  for (let lvl = 0; lvl < levels.length - 1; lvl++) {
    const nodes = levels[lvl];
    const isRightChild = (idx & 1) === 1;
    const sibIdx = isRightChild ? idx - 1 : idx + 1;
    const sibling = sibIdx < nodes.length ? nodes[sibIdx] : nodes[idx];
    path.push({ sibling, right: !isRightChild });
    idx >>= 1;
  }
  return path;
}

/** Recombine leaf + path → root. */
export function rootFromProof(
  encodedEvent: Uint8Array,
  path: readonly MerkleStep[],
): Uint8Array {
  let h = leafHash(encodedEvent);
  for (const step of path) {
    h = step.right ? nodeHash(h, step.sibling) : nodeHash(step.sibling, h);
  }
  return h;
}
