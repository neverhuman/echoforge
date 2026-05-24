# Multi-Agent Coordination Protocol (EchoForge)

EchoForge is built by multiple autonomous agents working in parallel. To stay
out of each other's way without a synchronous lock, agents coordinate via
a local-only, gitignored coordination scratchpad using the protocol described
here.

This document is the source of truth. The local scratchpad is the live state.

## Files

- Local coordination scratchpad — local-only and gitignored. Carries live work-slot
  state, blockers, handoffs, and an append-only message log. Two agents on
  the same machine see it; agents on different machines do not.
- `docs/comms-protocol.md` (this file) — committed spec. New agents read it
  on day one.
- `AGENTS.md` — agent rules (rtk prefix, ownership boundaries, receipt
  discipline, strict-open promise). Read first.
- `.agents/receipts/<slice>/<UTC-timestamp>.md` — the durable, committed
  record of what each agent actually did.

## Sections of the Local Scratchpad

Codex seeds the local scratchpad with three free-form sections: `## Completed`,
`## Pending`, `## Notes`. These are preserved as the informal log. The
protocol below appends four structured sections after `## Notes`:

1. `## Work Slots` — table of packets, owners, status, deps.
2. `## Blockers` — bullet list of cross-packet impediments.
3. `## Handoffs` — directed messages from one agent to another, dated.
4. `## Messages` — append-only chronological log of intent and decisions.

## Work-slot status values

- `open` — anyone may claim.
- `claimed:<agent>` — reserved but not started yet.
- `in-progress:<agent>` — actively being worked.
- `blocked:<agent>` — work started, paused on a dependency or external answer.
- `done` — receipt exists at the declared `receipt_dir`.
- `abandoned:<agent>` — released back to `open` with a Message explaining why.

## Atomic claim

To claim an `open` slot, edit the single row to change `status: open` →
`status: claimed:<your-agent-name>` and fill `started_utc` with
`date -u +%Y%m%dT%H%M%SZ`. Commit/save immediately. The unit of merge
conflict is the row, so two agents claiming the same row in parallel will
get a clean Git conflict on a single line.

## Conflict resolution: quote-and-respond

If you arrive and find a slot you wanted is already `claimed:<other>`, do
NOT overwrite. Append to `## Messages`:

```
> 20260518T0930Z other: claimed schemas-fill
20260518T0935Z me: I had drafted schemas — happy to hand notes over.
Releasing my claim. Picking up validate-crate instead.
```

If you arrive and find a conflicting in-progress edit to a file you own,
the latest commit wins, but the loser MUST quote the prior change in
`## Messages` and explain the response.

## Handoff format

Handoffs are explicit and dated:

```
- <UTC-ts> <from-agent> → <to-agent>: <short, actionable directive>.
  Files affected: <paths>. Receipts: <paths>.
```

A handoff is consumed by the receiving agent flipping the corresponding
work-slot row (claim, release, or mark blocked) and posting back to
`## Messages`.

## Append-only Messages

Newest at the bottom. Format:

```
- <UTC-ts> <agent>: <message>.
```

Never delete or rewrite a Message. Corrections are new Messages that
quote-respond to the prior.

## Receipt requirement

Every slot transitioning to `done` MUST have a receipt under
`.agents/receipts/<slice>/<UTC-timestamp>.md` matching the receipt-guard
template:

```
# <slice> Receipt
## Files Changed
- path/to/file: short purpose
## Commands Run
- rtk <cmd>: result
## Validation Results
- <lane>: pass|fail|skip
## Notes
- risks, follow-ups, blockers
```

Receipt filename: `^\d{8}T\d{6}Z?(?:-\d{4})?\.md$` (both `Z` and numeric
offset accepted). Validated by `node tools/receipt_guard.mjs --all`.

## Authorization scope

The approved Claude plan file for the current session (when accepted by
the user via Claude Code's plan mode) is the authorization for the work
slots in its packet list. Agents may take the listed actions without
per-zone re-confirmation. Anything outside the plan still falls under
`AGENTS.md` zone restrictions.

## Naming hygiene

- Agent names lower-case, no spaces: `claude`, `codex`, `claude-2`, etc.
- Slot names lower-case kebab: `governance-hardening`, `validate-crate`.
- UTC timestamps with `Z` suffix or numeric offset: `20260518T083350Z` or
  `20260518T022130-0600`.

## When to read this file again

- Every time you join a session.
- Before claiming a slot.
- Whenever `## Messages` references a behavior change.
