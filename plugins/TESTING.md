# ElizaOS Plugin Testing Guide

This document defines how plugins in this repository are tested, and — just
as importantly — how they are **not** tested. It exists because an earlier
iteration of these plugins shipped dead `__mocks__/` directories that looked
like test infrastructure but were never wired up. They were removed in the
same change that added this guide. Read this before adding test scaffolding.

---

## The three layers

Every plugin's test surface is split into three layers. Keep them separate.

| Layer | What it proves | Where it lives | When it runs |
|---|---|---|---|
| **Unit** | Plugin logic in isolation, given controlled inputs | `src/__tests__/*.test.ts` (vitest) | Every CI run |
| **Type** | Plugin compiles against real `@elizaos/core` source | `tsconfig.json` `paths` mapping + `bun typecheck` | Every CI run |
| **Live proof** | End-to-end against real `AgentRuntime` + real network/API | `scripts/*.ts` (manually invoked) | On demand, never in CI |

Unit tests **never** hit the network, a database, or a real `AgentRuntime`.
Live-proof scripts **always** do. Do not blur these.

---

## Unit tests: use inline structural fakes, not `__mocks__/`

### ✅ Do this

Build fakes **inline in the test file** and inject them via constructor or
function parameter. Import **real types** from `@elizaos/core` so the compiler
checks your fake's shape against the actual interface.

```ts
// src/__tests__/service.test.ts
import type { IAgentRuntime, Task } from "@elizaos/core";
import { MyService } from "../service.ts";

function makeFakeRuntime(overrides: Partial<...> = {}) {
  const tasks = new Map<string, Task>();
  return {
    logger: { info: vi.fn(), warn: vi.fn(), error: vi.fn(), debug: vi.fn() },
    getSetting: (k: string) => overrides.settings?.[k],
    createTask: async (t) => { tasks.set(t.id, t); return t; },
    getTasks: async ({ tags }) => [...tasks.values()].filter(/* real filter */),
    // ... only the methods the code-under-test actually calls
  } as unknown as IAgentRuntime;
}

it("transitions on terminal", async () => {
  const rt = makeFakeRuntime({ settings: { FOO: "bar" } });
  const svc = await MyService.start(rt);
  // ...
});
```

**Why this is the required pattern:**

- **Visible** — anyone reading the test sees exactly what is faked, in the
  same file. Nothing is hidden behind module-resolution magic.
- **Per-test customization** — `makeFakeRuntime({ settings: {...} })` lets
  each test set up its own world. A module-level mock is global.
- **Type-safe drift detection** — because you import the real `Task`,
  `IAgentRuntime`, etc., the compiler fails your fake when core changes shape.
  This is your TDD "red" signal for contract breaks.
- **No config** — works with bare `vitest run`; no `vitest.config.ts` needed.

### ❌ Do NOT do this

Do **not** create `src/__tests__/__mocks__/<package>.ts` files to replace
`@elizaos/core` or sibling plugins at the module level.

**Why this is banned:**

1. **It doesn't work without `vi.mock()`** — vitest does not auto-load
   `__mocks__/` directories. You must call `vi.mock("@elizaos/core")` in every
   test file. The deleted mocks had no such call, so they were silently dead.
2. **Scoped-package paths are easy to get wrong** — vitest looks for
   `__mocks__/@elizaos/core.ts` (scope as subdirectory), not
   `__mocks__/elizaos-core.ts`. The deleted mocks used the wrong convention.
3. **It erases type safety** — the deleted mocks declared
   `export type IAgentRuntime = unknown`. If active, every
   `runtime.anything(...)` call would typecheck. That defeats the point.
4. **It is not needed** — both plugin `tsconfig.json` files map
   `@elizaos/core` to `packages/typescript/src/index.node.ts` (source, not a
   built artifact). Bun and vitest resolve TS source directly, so the real
   package is always importable in tests. There is no "can't import core in
   test env" problem to solve.
5. **It hides what's faked** — a reader of `service.test.ts` cannot tell that
   `Service` is a stub without finding the `__mocks__/` file. Inline fakes are
   self-documenting.

If you think you need `vi.mock()`, you almost certainly want dependency
injection instead. The one acceptable global stub is `vi.stubGlobal("fetch", ...)`
for testing HTTP error-mapping paths that cannot be reached otherwise — and
even then, restore it in `afterEach`.

---

## What to fake, and where to draw the line

| Boundary | Fake it? | How |
|---|---|---|
| `IAgentRuntime` (DB, tasks, settings, logger) | ✅ Yes | Inline structural fake with in-memory `Map` stores |
| Sibling plugin services (e.g. `ScryptedAIService`) | ✅ Yes | Inline fake object returned from `fakeRuntime.getService()` |
| `fetch` / network | ✅ Yes | `vi.stubGlobal("fetch", ...)` or inject a fake client via DI |
| Clock (`setTimeout`) | ✅ Yes, for polling/backoff tests | `vi.useFakeTimers()` + `vi.runAllTimersAsync()` |
| `@elizaos/core` types & base classes | ❌ **Never** | Import the real thing via tsconfig `paths` |
| Pure plugin logic (adapters, parsers, FSM transitions) | ❌ **Never** | That's the code under test |

**Rule of thumb:** fake *infrastructure* (I/O, time, external processes), not
*types* or *logic*. If your fake has `export type X = unknown` in it, you've
crossed the line.

---

## Live-proof scripts (`scripts/*.ts`)

Unit tests prove "the plugin behaves correctly given these inputs." They do
**not** prove "the external API actually returns those inputs." That gap is
covered by live-proof scripts:

- They boot a real `AgentRuntime` with a real character.
- They read real secrets from `.env` (e.g. `SCRYPTEDAI_BEARER_TOKEN`).
- They make real network calls and wait real wall-clock time.
- They cost real money per run.

Therefore:

- They live in `scripts/`, **not** `__tests__/`, and are excluded from `vitest`.
- They are run manually: `bun scripts/prove.ts`.
- They are **not** part of CI. CI must be hermetic, fast, and free.
- They should assert observable outcomes (e.g. "a memory with an image
  attachment was delivered to the room") rather than internal state.

Do not move live-proof logic into vitest "to get more coverage." Coverage of
network-dependent paths in CI is false confidence.

---

## Historical note: the deleted `__mocks__/` directories

The following files were removed because they were provably dead (no
`vi.mock()` call, no `vitest.config.*`, wrong filename convention for scoped
packages) and would have *weakened* type safety if ever activated:

- `plugin-avb/src/__tests__/__mocks__/elizaos-core.ts`
- `plugin-avb/src/__tests__/__mocks__/plugin-scryptedai.ts`
- `plugin-scryptedai/src/__tests__/__mocks__/elizaos-core.ts`

They were created during early bootstrapping when `@elizaos/core` was thought
to be un-importable from source (a comment claimed "requires generated
protobuf files"). Once tsconfig `paths` mapped core to its source entry point,
the real package became importable, the author switched to inline fakes, and
the `__mocks__/` scaffolding was orphaned. All 143 tests pass identically with
them removed.

**Do not recreate them.** If you find yourself writing a `__mocks__/` file for
an `@elizaos/*` package, stop and use an inline structural fake instead.
