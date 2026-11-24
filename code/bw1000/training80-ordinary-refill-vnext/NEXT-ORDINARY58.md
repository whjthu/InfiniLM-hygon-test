# 2026-11-25 ordinary 58-successor calendar

The new wave is `training80-vnext-20261125-0841-ordinary58`. It uses the frozen 58-row successor evidence from the 05:44 wave and its pinned private source manifest. The calendar reader checks seven original two-point receipts, thirteen original terminal receipts, the exact historical 70-node calendar and the 80-node post-wave scheduler snapshot. It keeps the two original unresolved bridge requests and all three of their nodes excluded.

Preparation starts at 08:41:03.651 Beijing time. The candidate predecessor budgets range from 08:56:03.651 through 09:24:39.022. The offline partition has 16 parent-local groups, with 24 candidates from parent 1103606 and 34 from parent 1194900. Across the same 80 nodes and 640 cards, the 22 excluded nodes comprise 7 pre-bridge failures, 3 bridge-unknown nodes, 2 approval exclusions, 5 existing protections and 5 single-node scopes. This plan adds zero nodes and zero cards.

## Offline review

```powershell
$calendarFile = (Resolve-Path -LiteralPath 'local-only/training80-0544-ordinary-58-node-verified-standby-v1.json').Path
python -B -X utf8 code/bw1000/training80-ordinary-refill-vnext/coordinator.py check-calendar --calendar $calendarFile --calendar-sha 56da79542a9bb1181385e66f68ac9523f32c323b3cd31cb73a530283a329c83f
python -B -X utf8 code/bw1000/training80-ordinary-refill-vnext/check_calendar58_local.py local-only/training80-ordinary58-independent-absent.json
```

The offline calendar command and dispatch share the same source gate. The test uses the real 58-node source and the actual production dispatch function with local captured terminal and transport fixtures. It verifies an incomplete group, rejection of the old calendar and mismatched digest, a conflicting existing claim, and a repeated unknown request. The ongoing vNext ledger is untouched.

## Future operation

The resource owner can adopt this exact wave after the previous ordinary coordinator has released ownership. Provide a new pinned private adoption receipt with schema `training80.ordinary.vnext-sole-adoption.v1`, the wave name, `calendar_SHA256`, the independently reproduced `group_roster_SHA256`, `ledger_absolute`, `ledger_device`, `ledger_inode`, and true values for `previous_ordinary_controller_released` and `sole_current_coordinator_authorized`. Use the already existing fixed vNext ledger and verify its actual file identity. The receipt is an operator authorization and is outside this delivery.

When authorized, use the usual `dispatch` mode with the exact calendar path and SHA, the new wave name, group number, explicit request ID and pinned `--adoption` and `--adoption-sha` values. The unchanged original observer must show actual clean terminal state for every predecessor in the group before it can claim the group. An incomplete group writes zero claims. It can be observed again in the original window. Existing request IDs return the existing state before any publication or observation. A conflicting member blocks the entire group. The original device admission, durable bridge-intent boundary and UNKNOWN disposition still apply.
