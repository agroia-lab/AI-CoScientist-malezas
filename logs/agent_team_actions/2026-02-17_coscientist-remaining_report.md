# AI-CoScientist Remaining Steps Report

**Date:** 2026-02-17
**Team:** coscientist-remaining
**Baseline:** commit `a7193a0` (5-phase hardening complete, 45 tests)

---

## Team Composition

| Agent | Type | Task |
|-------|------|------|
| elo-tournaments | general-purpose | Extract elo.py + round-robin/swiss tournaments |
| prompts-config | general-purpose | Make prompts configurable |
| docs-writer | general-purpose | Create CONTRIBUTING.md |
| protocol-agent | general-purpose | Add AgentInterface protocol |
| team-lead | team-lead | Verification, lint fixes, report |

---

## Task 1: Extract elo.py + Tournament Modes (elo-tournaments)

**Status:** COMPLETED

### elo.py module
Created `ai_coscientist/elo.py` (117 lines) with:
- `calculate_elo_update(rating, opponent_rating, win, k_factor=32) -> int` — pure Elo calculation
- `generate_random_pairs(n, rounds_per_hypothesis=3, seed=None)` — current random matchmaking
- `generate_round_robin_pairs(n)` — every pair competes exactly once
- `generate_swiss_pairs(hypotheses_ratings, seed=None)` — pairs similar-rated hypotheses
- `swiss_rounds(n) -> int` — calculates optimal round count
- `validate_tournament_mode(mode)` — validates mode string

### Tournament modes in main.py
- Added `tournament_mode: str = "random"` parameter to `__init__`
- `_run_tournament_phase` dispatches to the correct pairing strategy
- Options: `"random"` (default, backwards compatible), `"round_robin"`, `"swiss"`

### Tests: `tests/test_elo.py` (25 tests)
- Elo calculation correctness, symmetry, custom k-factor
- Hypothesis.update_elo delegation to calculate_elo_update
- Random pairs: count, valid indices, determinism with seed
- Round-robin: pair count n*(n-1)/2, uniqueness, full coverage
- Swiss: adjacent rating pairing, odd count handling
- Mode validation, framework integration

---

## Task 2: Configurable Prompts (prompts-config)

**Status:** COMPLETED

### prompts.py changes (538 lines, +113/-17)
- Added `load_prompt(custom, default_fn)` helper:
  - `None` -> built-in default
  - File path -> reads file contents
  - Non-empty string -> returns as-is
- Refactored all 8 prompt functions to accept `custom_prompt: Optional[str] = None`
- Internal defaults moved to `_default_*_prompt()` private functions

### main.py changes
- Added `custom_prompts: Optional[Dict[str, str]] = None` parameter to `__init__`
- `_init_agents()` passes matching custom prompts to each `get_*_prompt()` call
- Keys: "generation", "reflection", "ranking", "evolution", "meta_review", "proximity", "tournament", "supervisor"

### Tests: `tests/test_prompts.py` (40 tests)
- load_prompt: None, empty, whitespace, custom string, file path, nonexistent file
- All 8 prompt functions: default non-empty, custom string override, None returns default, file path override
- Framework integration: accepts custom_prompts, passes them through

---

## Task 3: CONTRIBUTING.md (docs-writer)

**Status:** COMPLETED

Created `CONTRIBUTING.md` (2.9KB) with:
- Development setup (clone, install, .env)
- Running tests (pytest)
- Code style (line length 70, black, ruff, mypy)
- Project structure (7 modules described)
- PR process (fork, branch, tests, commit messages)
- Architecture notes (JSON output format, prompt/parsing coupling)

---

## Task 4: AgentInterface Protocol (protocol-agent)

**Status:** COMPLETED

### protocols.py (25 lines)
```python
@runtime_checkable
class AgentInterface(Protocol):
    agent_name: str
    def run(self, input: str) -> str: ...
```

### main.py changes
- Agent attributes type-hinted with `AgentInterface`
- Added `@classmethod from_custom_agents(cls, agents, **kwargs)` factory
  - Accepts `Dict[str, AgentInterface]` with role keys
  - Validates all 8 roles present
  - Validates each agent satisfies the protocol
  - Forwards remaining kwargs to `__init__`

### __init__.py
- Exported `AgentInterface`

### Tests: `tests/test_protocols.py` (12 tests)
- Stub agent satisfies protocol
- Bad agent (missing run) does not satisfy
- Plain object fails
- Protocol is runtime-checkable
- from_custom_agents: creates framework, assigns agents, missing role raises, bad agent raises, kwargs forwarded, defaults work
- AgentInterface importable from package

---

## Summary

| Metric | Before | After |
|--------|--------|-------|
| Test count | 45 | 122 |
| Test time | 0.09s | 0.16s |
| Modules | 5 | 7 (+elo.py, +protocols.py) |
| Prompt customization | None | File/string/default |
| Tournament modes | random only | random, round_robin, swiss |
| Agent abstraction | Locked to swarms.Agent | AgentInterface protocol |
| CONTRIBUTING.md | None | Yes |
| Ruff | Passing | Passing |
| Black | Passing | Passing |

### New module structure
```
ai_coscientist/
    __init__.py      (48 lines)   — re-exports
    elo.py           (117 lines)  — Elo rating + pairing strategies
    json_parser.py   (119 lines)  — JSON extraction
    main.py          (1593 lines) — framework orchestrator
    prompts.py       (538 lines)  — configurable agent prompts
    protocols.py     (25 lines)   — AgentInterface protocol
    types.py         (210 lines)  — data structures
```

### Files changed
```
 ai_coscientist/__init__.py |   4 +
 ai_coscientist/main.py     | 330 +++/---
 ai_coscientist/prompts.py  | 113 +++/---
 ai_coscientist/types.py    |  10 +-
 9 files changed, 629 insertions(+), 243 deletions(-)
```

### New files
```
 ai_coscientist/elo.py       (117 lines)
 ai_coscientist/protocols.py  (25 lines)
 CONTRIBUTING.md               (2.9KB)
 tests/test_elo.py            (257 lines)
 tests/test_prompts.py        (120 lines)
 tests/test_protocols.py      (190 lines)
```
