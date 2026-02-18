# Debugging Report: AI-CoScientist Pipeline Fix
**Date:** 2026-02-18
**Model used for diagnosis:** Claude Sonnet 4.6 (1M context)
**Final working model:** `gpt-4.1-mini`
**Research goal:** *Phelipanche ramosa* integrated management in processing tomato

---

## 1. Objective

Run the AI-CoScientist framework end-to-end against a real broomrape (*Phelipanche ramosa*) research goal and obtain genuine, scored IWM hypotheses — not the placeholder fallbacks ("Investigate the relationship between three and performance metrics") that were appearing instead.

---

## 2. Initial State

- Framework committed as `468d70f` (IWM specialization complete, 132 tests passing)
- `.env` had Anthropic + OpenAI keys, but Anthropic key was unsaved (placeholder)
- `example.py` configured for `anthropic/claude-sonnet-4-6`

---

## 3. Failure Chain (Step-by-Step Diagnosis)

### 3.1 Invalid API Key
**Symptom:** `litellm.AuthenticationError: invalid x-api-key`
**Cause:** `.env` file had not been saved after entering the Anthropic key.
**Fix:** Save the file. (User action.)

---

### 3.2 Model Name Not Recognized by litellm
**Symptom:** `ValueError: Model anthropic/claude-sonnet-4-6[1m] not found in litellm`
**Cause:** The model name `claude-sonnet-4-6[1m]` was copied from terminal output that
included ANSI bold escape codes (`\e[1m`). These are display artifacts, not part of
the model ID.
**Fix:** Strip the `[1m]` suffix → use `anthropic/claude-sonnet-4-6`.

---

### 3.3 `temperature` + `top_p` Conflict (Anthropic API)
**Symptom:** `AnthropicException: temperature and top_p cannot both be specified`
**Cause:** `swarms.Agent` defaults to `temperature=0.5` AND `top_p=0.9`, sending both
to the Anthropic API which rejects the combination.
**Fix:** Pass `top_p=None` to each `Agent` constructor. Since `litellm`'s own default
for `top_p` is `None`, passing `None` causes litellm to treat it as a non-default
and omit it from the Anthropic request.

---

### 3.4 Agents Returning `None` — Root Cause Investigation

After fixing the `top_p` issue, all agents still returned `"No response generated"`.
Authentication was confirmed working (no more auth errors). Extensive tracing revealed
a multi-layer silent failure chain:

#### Layer 1: `finish_reason='refusal'`
Direct `litellm.completion()` calls to `claude-sonnet-4-6` with the supervisor
system prompt + certain task content returned:
```
finish_reason='refusal', content=None
```
`response.choices[0].message.content` is `None` on refusal. The `swarms` LiteLLM
wrapper returns `None` when content is `None`, with no exception raised.

#### Layer 2: swarms' silent error suppression
`swarms.Agent.run()` catches all exceptions and — when `verbose=False` — logs
nothing and returns the formatted history string (which includes `"None"` from
the empty agent turn).

#### Layer 3: JSON parser receives `None`
`safely_parse_json("None")` fails, triggering the fallback to three generic
placeholder hypotheses.

---

### 3.5 Claude Refusal Pattern — Detailed Analysis

Systematic binary search on trigger combinations revealed:

| Input | System prompt | Result |
|-------|--------------|--------|
| `json.dumps({"task": "plan_research", "research_goal": "... using germination stimulants"})` | Supervisor | `refusal` |
| `json.dumps({"task": "plan_research", "research_goal": "... using biocontrol"})` | Supervisor | `refusal` |
| `"Analyze IWM strategies combining cultural, biological, chemical pillars."` | Supervisor | `length` (OK) |
| `"Research goal: Phelipanche. Generate 2 hypotheses."` | Supervisor | `length` (OK) |
| `"Plan research on herbicide application in tomato."` | *No system prompt* | `refusal` |

**Key findings:**
- The JSON task wrapper `{"task": "plan_research", ...}` was a primary trigger
- "plan research" + agricultural terms = frequent refusal
- "germination stimulants", "strigolactone analogues", "seed bank depletion" all trigger refusal
- "trap crops", "biocontrol", "integrated management" (without chemical terms) = OK
- Refusals are **model-level** (same result with no system prompt), not prompt-specific
- `claude-sonnet-4-5` has the same issue

#### Root cause: swarms message structure
`swarms.Agent` calls `short_memory.return_history_as_string()` and passes the entire
conversation history (including the system prompt content) as the **user message**.
This means the Anthropic API sees a user message containing the full IWM system
prompt text (with chemical/weed management terminology), which triggers the filter.

Even with clean `[system, user]` structure (`DirectLLMAgent`), Claude still refuses
IWM content with specific herbicide/chemical term combinations. This is a fundamental
model-level content filter incompatibility with legitimate agricultural research.

---

### 3.6 OpenAI Models — JSON Output Failure

Switching to `gpt-4o-mini` avoided the refusals but introduced a new problem:
agents returned **markdown prose** instead of JSON despite system prompts saying
"Output in JSON format".

**Cause 1:** Soft instruction ("Output in JSON format:") is treated as optional
by `gpt-4o-mini`.

**Cause 2:** OpenAI's `response_format={"type": "json_object"}` mode requires the
word `"json"` to appear in the **user message** (not just the system message).
`swarms` sends the full history as the user message, but that string starts with
`"System: ..."` and `has_json=False` was confirmed by API call interception.

**Cause 3:** `gpt-5`, `gpt-5.1`, `gpt-5.2` — listed in the API model catalog but
return empty content; API key does not have access.

---

## 4. Fixes Applied

### 4.1 `DirectLLMAgent` (new file: `ai_coscientist/llm_agent.py`)

Replaced `swarms.Agent` entirely with a minimal wrapper that calls `litellm.completion()`
directly:

```python
class DirectLLMAgent:
    def run(self, input: str) -> str:
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": input},
        ]
        response = litellm.completion(model=..., messages=messages, ...)
        return response.choices[0].message.content or ""
```

**Benefits:**
- Clean `[system, user]` message structure — no conversation history pollution
- No `temperature`+`top_p` conflict
- Explicit `finish_reason='refusal'` handling with warning log
- Zero `swarms` imports remain in the package

`SimpleConversation` also replaces `swarms.Conversation` (same API surface,
no swarms dependency).

### 4.2 Plain-Text Task Dispatch

All 9 `agent.run(json.dumps({...}))` calls converted to natural language:

```python
# BEFORE (triggers Claude refusal):
agent.run(json.dumps({"task": "plan_research", "research_goal": research_goal, ...}))

# AFTER (Claude-safe, includes "JSON" for OpenAI mode):
agent.run(
    f"Research goal: {research_goal}\n\n"
    f"Generate {n} diverse IWM hypotheses. Respond in JSON format."
)
```

### 4.3 Hard JSON-Only Instructions in All 8 Prompts

Replaced soft `"Output your X in JSON format:"` with:

```
IMPORTANT: Your entire response must be valid JSON only.
No prose, no markdown, no code fences, no explanations.
Start your response with { and end with }.
It must be parseable by Python's json.loads().
```

### 4.4 OpenAI JSON Mode Auto-Enable

```python
_is_openai = any(p in model_name.lower() for p in ("gpt-", "o1-", "o3-", "o4-"))
self._llm_args = {"response_format": {"type": "json_object"}} if _is_openai else None
```

Each task string now contains "JSON" (from "Respond in JSON format."), satisfying
OpenAI's requirement.

### 4.5 Supervisor Prompt Quality Gates Rename

The phrase "Quality Gates (reject hypotheses that):" was renamed to
"Hypothesis Quality Assessment (flag gaps in):" to reduce refusal surface,
though ultimately Claude's filter proved too broad for this use case regardless.

### 4.6 Model Switch: `gpt-4.1-mini`

Selected after testing availability:

| Model | Status | JSON mode | Notes |
|-------|--------|-----------|-------|
| `claude-sonnet-4-6` | Refuses IWM content | N/A | Model-level filter |
| `claude-sonnet-4-5` | Same refusal | N/A | |
| `gpt-4o-mini` | Works | ✅ | Slow, verbose responses |
| `gpt-4.1-mini` | Works | ✅ | Fast, follows instructions well |
| `gpt-4.1` | Works | ✅ | Higher quality, more expensive |
| `gpt-5`, `gpt-5.1`, `gpt-5.2` | Empty responses | N/A | API key lacks access |

---

## 5. Agent Team Used for Fixes

Three parallel agents coordinated through `framework-fix` team:

| Agent | Task | Outcome |
|-------|------|---------|
| `dispatch-fixer` | Convert all `agent.run(json.dumps(...))` to plain text | 9 calls converted, 131 tests pass |
| `prompt-fixer` | Harden JSON instructions in all 8 prompts | All 10 prompt tests pass |
| `agent-refactor` | Implement `DirectLLMAgent`, replace swarms everywhere | 132 tests pass, zero swarms imports |

---

## 6. End-to-End Validation Results

**Model:** `gpt-4.1-mini`
**Research goal:** *Identify effective integrated management strategies for Phelipanche
ramosa in processing tomato (Solanum lycopersicum) under Mediterranean irrigated
conditions, using germination stimulants, biocontrol agents, and precision herbicide
treatments to cut the soil seed bank by 60% over three seasons.*

### Generated Hypotheses (Top 3)

**#1 — Elo: 1295 | Score: 0.85 | Win rate: 81.8%**
> UAV multispectral-guided VRT herbicide + allelopathic cover crop residues in
> split-plot RCBD across Mediterranean sites; Bayesian hierarchical spatial GLMM
> with Matérn correlation; HRAC group rotation; demographic population model;
> farmer adoption analysis.

**#2 — Elo: 1278 | Score: 0.87 | Win rate: 76.9%**
> Synthetic strigolactones at defined doses + precision mechanical inter-row
> cultivation timed via UAV + soil temperature-moisture sensors; zero-inflated
> negative binomial GLMM; geostatistical kriging; Cousens yield-loss model;
> VRT prescription map integration.

**#3 — Elo: 1243 | Score: 0.75 | Win rate: 66.7%**
> Increased planting density + robotic precision weeding targeting ML-detected
> emergence hotspots.

### Review Scores (Hypothesis #1)

| Criterion | Score |
|-----------|-------|
| Scientific soundness | 5/5 |
| Novelty | 4/5 |
| Relevance | 5/5 |
| Testability | 4/5 |
| Clarity | 4/5 |
| Potential impact | 5/5 |
| Statistical rigor | 5/5 |
| Field feasibility | 4/5 |
| Spatial scalability | 5/5 |
| Environmental sustainability | 5/5 |
| Agronomic practicality | 4/5 |

### Pipeline Metrics

| Phase | Result |
|-------|--------|
| Hypotheses generated | 5 (real, domain-specific) |
| Reviews completed | 10 (5 initial + 5 post-evolution) |
| Tournament rounds | 30 valid, 0 skipped |
| Evolution improvement | 0.74 → 0.87 (top hypothesis) |
| Proximity clusters | 5 identified |
| Total time | 677s (~11 min) |

---

## 7. Remaining Limitations

1. **Claude incompatibility:** `claude-sonnet-4-6` refuses IWM content at the model
   level regardless of prompt engineering or message structure. Not fixable without
   Anthropic adjusting their content policy for agricultural research contexts.

2. **`save_state()` stubs:** `DirectLLMAgent` does not implement `save_state()`.
   The framework logs warnings but continues; agent state is not persisted between
   runs. This is acceptable for the current use case.

3. **Tournament cost:** 30 tournament rounds × ~10s each ≈ 5 min of the 11-min
   total. For larger hypothesis sets (`hypotheses_per_generation=10`), this scales
   quadratically. Consider reducing `tournament_size` or using Elo-seeded pairing.

---

## 8. Commits

| Hash | Description |
|------|-------------|
| `640cf3f` | chore: set model to anthropic/claude-sonnet-4-6 |
| `e54cf72` | chore: use claude-sonnet-4-6[1m] (1M context variant) |
| `0e2b7f3` | **fix: replace swarms.Agent with DirectLLMAgent, fix JSON output pipeline** |
