/**
 * Dynamic difficulty adjustment.
 *
 * Goal: hold mean block time at TARGET_BLOCK_MS (60s) given a fixed CPU
 * allocation (~5%). The miner's hashrate is unknown a priori (depends on
 * host hardware) and can drift (thermal throttling, host load), so we
 * retarget continuously.
 *
 * Bitcoin retargets every 2016 blocks. We can't wait 2016 minutes to
 * correct a bad initial guess, so we use an EMA over the last few block
 * times. The EMA smooths out per-block variance (PoW solve time is
 * geometrically distributed — high variance is intrinsic) while still
 * tracking real hashrate drift within a few blocks.
 *
 * The adjustment is multiplicative on the *target*, not on a "difficulty
 * number" — keeps the math in one domain. We clamp the per-step factor
 * to avoid wild swings from a single lucky/unlucky solve.
 */

import { bitsToTarget, targetToBits } from "./encoding.ts";

// ----------------------------------------------------------------------------
// Tunables
// ----------------------------------------------------------------------------

/** One block per minute. The whole point. */
export const TARGET_BLOCK_MS = 60_000;

/**
 * Genesis difficulty. Chosen so that a low-end machine at 5% CPU finds
 * the first block in roughly the right ballpark (a few seconds to a few
 * minutes). The retargeter converges from there. Encoded as nBits.
 *
 * Mantissa 0x00ffff at exponent 0x1f gives a target with 1 leading zero
 * byte → expected ~256 hashes per solve. At ~5% of, say, 2M sha256d/sec
 * = 100k h/s effective, that's ~3ms. Way too easy — but the adjuster
 * will crank it up within a handful of blocks, and starting easy means
 * we never stall on first boot waiting forever for genesis.
 */
export const GENESIS_BITS = 0x1f00ffff;

/**
 * Hardest target we ever set. Prevents a runaway adjuster on a beefy
 * host from setting a target so hard that one unlucky streak takes an
 * hour. Mantissa 0x000001 at exponent 0x1b → ~24 leading zero bits.
 */
export const MIN_BITS = 0x1b000001;

/** EMA smoothing factor. 0.3 = ~70% weight on history, 30% on latest. */
const EMA_ALPHA = 0.3;

/** Per-step adjustment clamp. 4× max swing per block. */
const ADJUST_CLAMP_MIN = 0.25;
const ADJUST_CLAMP_MAX = 4.0;

// ----------------------------------------------------------------------------
// Adjuster state
// ----------------------------------------------------------------------------

export interface DifficultyState {
  bits: number;
  /** EMA of recent block solve times (ms). */
  emaBlockMs: number;
}

export function initialDifficulty(): DifficultyState {
  return { bits: GENESIS_BITS, emaBlockMs: TARGET_BLOCK_MS };
}

// ----------------------------------------------------------------------------
// Retarget
// ----------------------------------------------------------------------------

/**
 * Compute the next block's target given the just-solved block's solve time.
 *
 * If blocks are coming too fast (emaBlockMs < target), shrink the target
 * (harder). If too slow, grow it (easier). The factor is the ratio of
 * actual to desired time, clamped.
 *
 * @param state current difficulty state
 * @param lastSolveMs wall-clock time the just-solved block took
 */
export function retarget(
  state: DifficultyState,
  lastSolveMs: number,
): DifficultyState {
  // Update EMA. First few blocks will swing because the EMA hasn't
  // converged yet — that's fine, the clamp catches the worst of it.
  const ema = state.emaBlockMs * (1 - EMA_ALPHA) + lastSolveMs * EMA_ALPHA;

  // Adjustment factor for the *target* (not difficulty).
  // Too fast → factor < 1 → smaller target → harder.
  // Too slow → factor > 1 → larger target → easier.
  let factor = ema / TARGET_BLOCK_MS;
  if (factor < ADJUST_CLAMP_MIN) factor = ADJUST_CLAMP_MIN;
  if (factor > ADJUST_CLAMP_MAX) factor = ADJUST_CLAMP_MAX;

  const newBits = scaleTarget(state.bits, factor);
  return { bits: newBits, emaBlockMs: ema };
}

// ----------------------------------------------------------------------------
// Target arithmetic
// ----------------------------------------------------------------------------

/**
 * Multiply the 256-bit target by a positive factor and re-encode.
 *
 * We do the multiply in JS-number space on the most-significant 6 bytes
 * of the target (≈ 48 bits of precision, well within Number.MAX_SAFE).
 * Sub-percent error from truncation, which the EMA absorbs anyway.
 */
export function scaleTarget(bits: number, factor: number): number {
  const target = bitsToTarget(bits);

  // Find MSB.
  let msb = 0;
  while (msb < 32 && target[msb] === 0) msb++;
  if (msb === 32) {
    // All-zero target → impossibly hard. Reset to genesis.
    return GENESIS_BITS;
  }

  // Read up to 6 significant bytes.
  const len = Math.min(6, 32 - msb);
  let mantissa = 0;
  for (let i = 0; i < len; i++) mantissa = mantissa * 256 + target[msb + i];

  // Scale.
  let scaled = mantissa * factor;

  // Renormalize: if scaling overflowed our 6-byte window, shift right
  // and bump the exponent (move msb left). If it underflowed below 1
  // significant byte, shift left.
  let newMsb = msb;
  while (scaled >= 256 ** len && newMsb > 0) {
    scaled /= 256;
    newMsb--;
  }
  while (scaled < 256 ** (len - 1) && newMsb + len < 32) {
    scaled *= 256;
    newMsb++;
  }

  // Write back into a fresh 32-byte buffer.
  const out = Buffer.alloc(32);
  let m = Math.floor(scaled);
  for (let i = len - 1; i >= 0; i--) {
    out[newMsb + i] = m & 0xff;
    m = Math.floor(m / 256);
  }

  const newBits = targetToBits(out);

  // Floor at MIN_BITS (don't get harder than the cap). Comparing nBits
  // values directly is wrong (encoding isn't monotone), so compare the
  // expanded targets.
  const newTarget = bitsToTarget(newBits);
  const minTarget = bitsToTarget(MIN_BITS);
  for (let i = 0; i < 32; i++) {
    if (newTarget[i] < minTarget[i]) return MIN_BITS; // Harder than cap.
    if (newTarget[i] > minTarget[i]) break; // Easier — fine.
  }

  return newBits;
}
