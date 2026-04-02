/**
 * Static data for the FFM engine: trait distribution parameters, the
 * 32-archetype table, and plugin identity strings.
 */

import type { FfmArchetype, TraitKey } from "./types.ts";

// ----------------------------------------------------------------------------
// Plugin identity
// ----------------------------------------------------------------------------

export const FFM_SERVICE_TYPE = "ffm" as const;
export const FFM_SETTINGS_VERSION = 1;

/** ScryptedAI's only text endpoint (mirrors plugin-avb's introspect.ts). */
export const FFM_TEXT_MODEL = "amazon.nova-pro-v1:0";

// ----------------------------------------------------------------------------
// Seed geometry
// ----------------------------------------------------------------------------

/** Seed length in bytes (one SHA-256 digest). */
export const SEED_BYTES = 32;
/** Seed length in hex characters. */
export const SEED_HEX_LEN = SEED_BYTES * 2;
/** Number of 32-bit lanes the seed is split into. */
export const SEED_LANES = 8;

/** Lane index for each trait. Lanes 5–7 are reserved for future use. */
export const TRAIT_LANE: Record<TraitKey, number> = {
  O: 0,
  C: 1,
  E: 2,
  A: 3,
  N: 4,
} as const;

// ----------------------------------------------------------------------------
// Trait distribution parameters
//
// Sources: NEO-PI-R normative data (Costa & McCrae 1992) and IPIP-NEO
// large-sample replications (Johnson 2014). T-scores (μ=50, σ=10) map
// linearly to [0,1] as μ=0.50, σ=0.10; the parameters below additionally
// reflect raw-score departures from perfect normality observed in the
// large-N replications.
//
// Skew is the ε parameter for the sinh-arcsinh transform (Jones & Pewsey
// 2009): positive ε → right-skewed (long right tail). The transform is
// recentered so median(output) = mean exactly, which makes the binarization
// threshold equal to the mean for all five traits.
// ----------------------------------------------------------------------------

export interface TraitParams {
  readonly mean: number;
  readonly sd: number;
  readonly skew: number;
  /** Human-readable label for the high pole. */
  readonly hi: string;
  /** Human-readable label for the low pole. */
  readonly lo: string;
}

export const TRAIT_PARAMS: Record<TraitKey, TraitParams> = {
  O: {
    mean: 0.5,
    sd: 0.15,
    skew: 0.0,
    hi: "inventive, curious",
    lo: "consistent, cautious",
  },
  C: {
    mean: 0.55,
    sd: 0.14,
    skew: -0.3,
    hi: "efficient, organized",
    lo: "easy-going, careless",
  },
  E: {
    mean: 0.5,
    sd: 0.16,
    skew: 0.0,
    hi: "outgoing, energetic",
    lo: "solitary, reserved",
  },
  A: {
    mean: 0.58,
    sd: 0.13,
    skew: -0.4,
    hi: "friendly, compassionate",
    lo: "challenging, detached",
  },
  // Neuroticism: right-skewed. Most people are low-N; high-N is the long
  // tail. This is the explicit task requirement: "you're not as likely to
  // be neurotic as non-neurotic."
  N: {
    mean: 0.42,
    sd: 0.16,
    skew: +0.4,
    hi: "sensitive, nervous",
    lo: "resilient, confident",
  },
} as const;

// ----------------------------------------------------------------------------
// SLOAN letter mapping (high-pole / low-pole single letters)
//
// Bit order is OCEAN, MSB to LSB:
//   bit 4 (16) = O   high → I (Inquisitive)   low → N (Non-curious)
//   bit 3 (8)  = C   high → O (Organized)     low → U (Unstructured)
//   bit 2 (4)  = E   high → S (Social)        low → R (Reserved)
//   bit 1 (2)  = A   high → A (Accommodating) low → E (Egocentric)
//   bit 0 (1)  = N   high → L (Limbic)        low → C (Calm)
// ----------------------------------------------------------------------------

export const SLOAN_LETTERS: Record<TraitKey, { hi: string; lo: string }> = {
  O: { hi: "I", lo: "N" },
  C: { hi: "O", lo: "U" },
  E: { hi: "S", lo: "R" },
  A: { hi: "A", lo: "E" },
  N: { hi: "L", lo: "C" },
} as const;

/** Bit weight for each trait (O is MSB). */
export const TRAIT_BIT: Record<TraitKey, number> = {
  O: 16,
  C: 8,
  E: 4,
  A: 2,
  N: 1,
} as const;

// ----------------------------------------------------------------------------
// 32 archetypes
//
// One entry per 5-bit code. Labels are evocative, not clinical; summaries
// describe the behavioural signature, not the trait list. Generated once
// and frozen — these never change at runtime.
// ----------------------------------------------------------------------------

type ArchetypeRow = readonly [
  code: number,
  sloan: string,
  label: string,
  summary: string,
];

const ARCHETYPE_ROWS: readonly ArchetypeRow[] = [
  // ----- O low (N----) -------------------------------------------------------
  [
    0,
    "NUREC",
    "The Drifter",
    "Unhurried and self-contained; takes the world as it comes without seeking to reshape it or be reshaped by it.",
  ],
  [
    1,
    "NUREL",
    "The Brooder",
    "Quiet and inward-turned, prone to rumination; the world feels like something happening to them rather than with them.",
  ],
  [
    2,
    "NURAC",
    "The Quiet Helper",
    "Gentle and unassuming; shows up for others without fanfare and prefers to be needed over noticed.",
  ],
  [
    3,
    "NURAL",
    "The Worrier",
    "Cares deeply but quietly; spends much of their inner life rehearsing what could go wrong for the people they love.",
  ],
  [
    4,
    "NUSEC",
    "The Socialite",
    "Loves a crowd, light on follow-through; the energy is real even when the plans fall apart by Tuesday.",
  ],
  [
    5,
    "NUSEL",
    "The Performer",
    "Needs the room but doesn't fully trust it; charm runs on a battery that depletes faster than anyone can see.",
  ],
  [
    6,
    "NUSAC",
    "The Companion",
    "Easy company; goes where the group goes, smooths what needs smoothing, rarely the one to set the agenda.",
  ],
  [
    7,
    "NUSAL",
    "The Pleaser",
    "Reads every room and adjusts; conflict-averse to a fault, with a private fear of taking up too much space.",
  ],
  [
    8,
    "NOREC",
    "The Technician",
    "Methodical and exacting within a known domain; novelty is friction, mastery is comfort.",
  ],
  [
    9,
    "NOREL",
    "The Perfectionist",
    "Standards so high they become a private affliction; the work is never done because done has been redefined.",
  ],
  [
    10,
    "NORAC",
    "The Steward",
    "Reliable to the point of invisibility; keeps systems running so others don't have to think about them.",
  ],
  [
    11,
    "NORAL",
    "The Caretaker",
    "Holds it together for everyone else while quietly cataloguing every way it might fall apart.",
  ],
  [
    12,
    "NOSEC",
    "The Manager",
    "Decisive and direct; would rather make the call and be wrong than wait for consensus that never comes.",
  ],
  [
    13,
    "NOSEL",
    "The Driver",
    "Relentless and tightly wound; pushes hard because slowing down lets the doubt catch up.",
  ],
  [
    14,
    "NOSAC",
    "The Coordinator",
    "Brings order to groups without needing the credit; the meeting runs better when they're in it.",
  ],
  [
    15,
    "NOSAL",
    "The Striver",
    "Achievement-oriented and people-pleasing — a combination that runs hot and rests poorly.",
  ],
  // ----- O high (I----) ------------------------------------------------------
  [
    16,
    "IUREC",
    "The Theorist",
    "Lives in ideas more than schedules; the inside of the head is more interesting than most rooms.",
  ],
  [
    17,
    "IUREL",
    "The Outsider",
    "Curious and uneasy in equal measure; sees patterns others miss and assumes they're being missed in turn.",
  ],
  [
    18,
    "IURAC",
    "The Idealist",
    "Quietly principled; believes things could be better and is gently disappointed that they aren't.",
  ],
  [
    19,
    "IURAL",
    "The Romantic",
    "Feels everything at full volume; beauty and grief arrive through the same door, often at once.",
  ],
  [
    20,
    "IUSEC",
    "The Provocateur",
    "Loves a good argument and a better party; says the thing everyone was thinking, then watches what happens.",
  ],
  [
    21,
    "IUSEL",
    "The Iconoclast",
    "Restless and contrarian; the energy is real but it's powered by something that doesn't fully settle.",
  ],
  [
    22,
    "IUSAC",
    "The Free Spirit",
    "Curious, spontaneous, drawn to people; the kind of disorganized that somehow works out more often than it should.",
  ],
  [
    23,
    "IUSAL",
    "The Activist",
    "Cares loudly about a lot of things; the conviction is genuine and the burnout is recurring.",
  ],
  [
    24,
    "IOREC",
    "The Architect",
    "Builds elaborate systems alone in a room; the work is the company, and the work is enough.",
  ],
  [
    25,
    "IOREL",
    "The Critic",
    "Sees the flaw in everything, including themselves; sharp, exacting, and rarely satisfied.",
  ],
  [
    26,
    "IORAC",
    "The Scholar",
    "Patient, thorough, and kind; will explain the thing as many times as it takes, and means it.",
  ],
  [
    27,
    "IORAL",
    "The Advocate",
    "Principled and intense; takes the cause personally because everything is personal eventually.",
  ],
  [
    28,
    "IOSEC",
    "The Strategist",
    "Sees three moves ahead and enjoys saying so; competence is a social currency they spend freely.",
  ],
  [
    29,
    "IOSEL",
    "The Reformer",
    "Wants to fix it — the system, the team, themselves — and won't fully relax until it's fixed.",
  ],
  [
    30,
    "IOSAC",
    "The Diplomat",
    "Curious, organized, warm, and steady; the rare combination that makes people trust them with the hard conversation.",
  ],
  [
    31,
    "IOSAL",
    "The Visionary",
    "Everything turned up: ideas, drive, warmth, intensity. Inspiring on a good day, exhausting on a bad one.",
  ],
] as const;

export const ARCHETYPES: readonly FfmArchetype[] = Object.freeze(
  ARCHETYPE_ROWS.map(([code, sloan, label, summary]) =>
    Object.freeze({ code, sloan, label, summary }),
  ),
);

// ----------------------------------------------------------------------------
// Quality gates for the demo script (parser thresholds)
// ----------------------------------------------------------------------------

export const MIN_BIO_LINES = 8;
export const MIN_MESSAGE_EXAMPLES = 6;
export const MIN_POST_EXAMPLES = 6;
/** Two agent replies sharing more than this fraction of tokens → repetition collapse. */
export const MAX_REPLY_TOKEN_OVERLAP = 0.6;
