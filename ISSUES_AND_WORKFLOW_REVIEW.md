# Issues & Workflow Conceptual Review
**Date:** 2026-02-18
**Purpose:** Document all technical problems encountered + critical evaluation of the
AI-CoScientist methodology as applied to IWM research. Reference document for next session.

---

## Part I — Technical Issues

### T1. Environment / Setup

| Issue | Root Cause | Status |
|-------|-----------|--------|
| `.env` not saved after editing | User did not press Ctrl+S | Fixed (user action) |
| `claude-sonnet-4-6[1m]` invalid model ID | ANSI bold escape code `\e[1m]` copied from terminal output | Fixed: stripped suffix |
| `swarms 0.9.9` import conflict | langchain dependency clash | Fixed: upgraded to 9.0.0 |

---

### T2. Anthropic API Compatibility

**Problem:** `swarms.Agent` sends `temperature=0.5` AND `top_p=0.9` simultaneously.
The Anthropic API rejects this combination with HTTP 400.

**Fix:** Pass `top_p=None` to each Agent constructor so litellm treats it as the default
and omits it from the request.

---

### T3. Claude Content Filter Refusals (`finish_reason='refusal'`)

This was the most time-consuming issue. Claude `claude-sonnet-4-6` (and `claude-sonnet-4-5`)
refuses a wide range of legitimate agricultural research inputs. Key patterns found:

| Input pattern | Result |
|--------------|--------|
| `json.dumps({"task": "plan_research", "research_goal": "... herbicide"})` | `refusal` |
| `json.dumps({"task": "plan_research", "research_goal": "... germination stimulants"})` | `refusal` |
| `json.dumps({"task": "plan_research", "research_goal": "... biocontrol"})` | `refusal` |
| `json.dumps({"task": "plan_research", "research_goal": "Phelipanche management"})` | OK |
| `"Analyze IWM strategies combining cultural, biological, chemical pillars."` | OK |
| `"Research goal: X. Generate 2 hypotheses."` | OK |
| No system prompt + "Plan research on herbicide application in tomato." | `refusal` |

**Root cause — two layers:**

1. **Model-level filter**: `claude-sonnet-4-6` treats combinations of "plan research" +
   agricultural chemical terms as policy violations, regardless of system prompt context.

2. **swarms architecture**: `swarms.Agent` builds a long conversation history string and
   sends it entirely as the **user message** to the LLM API. When this string includes the
   system prompt (which contains IWM/herbicide terminology), the combined user message
   length and content triggers Claude's filter even on clean calls.

**Fix applied:** Replaced `swarms.Agent` with `DirectLLMAgent` (clean `[system, user]`
messages) + converted all task dispatch from `json.dumps({...})` to plain text.

**Residual issue:** Even with `DirectLLMAgent`, Claude `claude-sonnet-4-6` refuses the
reflection, ranking, and tournament agents when hypothesis texts contain herbicide content.
Claude is **not suitable for IWM/agricultural research** at this level of specificity
without explicit content policy exceptions from Anthropic.

---

### T4. Agents Returning Markdown Instead of JSON

**Problem:** Both `gpt-4o-mini` and `claude-sonnet-4-6` ignored system prompt instructions
saying "Output in JSON format:" and returned narrative markdown prose. The JSON parser
received content it could not parse and fell back to 3 generic placeholder hypotheses.

**Root causes:**
- Soft instruction ("Output in JSON format:") treated as optional guidance
- OpenAI's `response_format={"type": "json_object"}` requires the word "json" in the
  **user message** specifically — `swarms` does not guarantee this
- `swarms.Agent` returns `history_output_formatter(short_memory)` — the full conversation
  history string — not just the last agent response

**Fix:** Hard JSON-only instruction block in all 8 prompts + OpenAI JSON mode auto-enabled
via `llm_args` when model name contains `"gpt-"`.

---

### T5. GPT-5 Series Not Accessible

Tested: `gpt-5`, `gpt-5-mini`, `gpt-5.1`, `gpt-5.2`.
All return empty content (`finish_reason='length'`, content=`''`).
API key does not have access. Models are listed in the catalog but not available.

**Available models confirmed working:** `gpt-4.1-mini`, `gpt-4.1`, `gpt-4.1-nano`.

---

### T6. GitHub Push Issues

*To be documented in the next session. The repo was pushed successfully during this
session (`0e2b7f3`, `c39661f`) but recurring push issues exist that need investigation.*

---

### T7. `save_state()` Not Implemented

`DirectLLMAgent` does not implement `save_state()`. The framework logs 8 warnings
at the end of each run. Agent state is not persisted between runs.

**Impact:** Low for single-run use. Affects reproducibility and incremental workflows.
**Status:** Open.

---

## Part II — Conceptual Review of the AI-CoScientist Workflow

The following evaluates the **ideas and methodology** of the framework, not the code.

---

### W1. The Core Premise: Tournament-Based Hypothesis Evolution

**The paper's claim:** Pairwise tournament comparison + Elo ratings will surface the
best hypotheses through competitive selection, analogous to evolutionary pressure.

**Assessment for IWM research: Partially valid.**

Elo ratings assume a **total ordering** — that hypothesis A is objectively better than
B in a consistent, transitive way. For chess this holds. For scientific hypotheses, it
does not:

- H1 (strigolactone + biocontrol) may beat H3 (solarisation) in a head-to-head based
  on methodological novelty, but a farmer with no irrigation infrastructure would reverse
  that ranking
- The tournament agent picks a "winner" without specifying along what dimension
- 15 pairwise rounds with 5 hypotheses (~3 matches each) is too few for Elo to stabilise
  — the ratings carry large variance
- **The system optimises for what the LLM judge prefers, not for what a domain expert would rank**

**Recommendation:** Replace pairwise "winner" tournament with a **multi-criterion scoring
tournament** where the judge rates each hypothesis on 3-4 dimensions (scientific merit,
field realism, novelty) separately, and Elo is calculated per dimension.

---

### W2. The Generation Phase: Supervisor + Generator Redundancy

**Current design:** The supervisor generates a "research plan" (JSON), which is then
passed to the generation agent as context.

**Problem observed:**
- The supervisor's plan is embedded in the generation task as JSON context
- The generation agent largely ignores it and generates hypotheses based on the research
  goal directly
- Both agents receive the same system prompt framing (IWM pillars, statistical methods)
- The supervisor adds ~18 seconds of latency and ~$0.01 of cost with minimal impact on
  output quality

**Assessment:** The supervisor role may make sense at `max_iterations=5` with `N=20`
hypotheses, where it can meaningfully enforce pillar diversity across a large portfolio.
At the current settings (`N=5`, `iterations=1`) it adds overhead without benefit.

**Recommendation:** Make the supervisor optional, activated only when
`hypotheses_per_generation >= 10`. For small runs, pass the research goal directly
to the generation agent.

---

### W3. What "Hypothesis" Means in This Context

**The core conceptual problem:** The framework generates **research project proposals**,
not **scientific hypotheses**.

A scientific hypothesis is a precise, falsifiable statement: *"If X, then Y, because Z."*

What the framework generated:
> *"Sequential pre-season application of Nijmegen-1 (10⁻⁷ M via drip irrigation, 21 and
> 14 days before transplanting) combined with inundative soil drenching of F. oxysporum
> f.sp. orthoceras (10⁸ conidia mL⁻¹, 7 days post-stimulant) will reduce viable P. ramosa
> seed bank density by ≥60% over three consecutive seasons, outperforming either treatment
> alone, because suicidal germination commits broomrape seeds to radicle elongation in the
> absence of a host, rendering germinated seeds maximally vulnerable to fungal infection
> of the exposed radicle tissue before tubercle formation can occur..."*

This is a mini research protocol with mechanistic justification — useful, but not a
testable hypothesis in the strict sense. A focused hypothesis would be:

> *"Soil application of Nijmegen-1 (10⁻⁷ M) 14 days before F. oxysporum f.sp. orthoceras
> inoculation reduces P. ramosa seed bank density more than either treatment alone."*

**Implication:** The evolution step makes hypotheses **longer and more elaborate**, not
**more focused**. This is the opposite of what good scientific hypothesis refinement
should do. Real hypothesis evolution should prune, sharpen, and increase falsifiability.

**Recommendation:** Add an explicit "sharpening" instruction to the evolution prompt:
*"Reduce this hypothesis to its single most testable, falsifiable core claim. Remove
implementation details that belong in a research protocol, not a hypothesis."*

---

### W4. Review Criteria Mismatch

**Current 11 criteria:**
6 generic (soundness, novelty, relevance, testability, clarity, impact) +
5 agronomy-specific (statistical rigor, field feasibility, spatial scalability,
environmental sustainability, agronomic practicality).

**Problem observed:** The framework scored H1 and H2 at 0.87 and 0.85 respectively with
mostly 4-5/5 across all criteria. There is **very little score differentiation** — the
portfolio bunches around 0.72–0.87 with no hypothesis below 3/5 on any criterion.

**Why this happens:**
- The reflection agent (gpt-4.1-mini) is instructed to be a peer reviewer, but
  peer reviewers in this context are trained to be constructive rather than discriminating
- The model lacks the domain expertise to penalise scientifically wrong claims
  (e.g., Trichoderma as P. ramosa biocontrol agent receiving 4/5 on "scientific soundness")
- Scores cluster high because the generation prompt already filters for "good" hypotheses

**Recommendation:**
1. Add **adversarial review** — a second reflection agent tasked specifically to find
   flaws, not score merit
2. Add **domain expert calibration examples** to the reflection prompt showing what
   a 2/5 vs 5/5 hypothesis looks like for each agronomy-specific criterion
3. Separate review rounds: one round for scientific validity, one for practical feasibility

---

### W5. Evolution Optimises for Complexity, Not Quality

**Observation:** The evolution step took the two highest-Elo hypotheses and returned
versions that were **longer, more detailed, and more methodologically elaborate**.
Score improved from 0.74–0.78 to 0.85–0.87 — but this may reflect the reflection
agent rewarding detail, not actual scientific improvement.

**The pattern:**
- Pre-evolution: "UAV mapping + VRT imazamox → 60% seed bank reduction"
- Post-evolution: "UAV (MicaSense RedEdge, 5 cm GSD) + CNN trained on red-edge and NIR
  spectral signatures + prescription map + variable-rate at 0/25/50 g a.i. ha⁻¹ +
  Bayesian hierarchical spatial GLMM with Matérn correlation structure + geostatistical
  kriging + Cousens yield-loss model + VRT prescription map integration with FMIS..."

The evolved version is more like a research grant proposal. An agronomist would flag it
as **over-engineered for a first experimental test**.

**Recommendation:**
- Add a "parsimony constraint" to the evolution prompt: *"The refined hypothesis should
  be testable in a single field season with standard equipment. Remove components that
  require separate studies to validate."*
- Score hypotheses for conciseness as a separate dimension (a 2-sentence hypothesis
  that is clear and testable should rank higher than a 10-sentence one that says the same)

---

### W6. The Loop Structure Does Not Truly Iterate

**Stated design:** Multi-iteration refinement — hypotheses improve across iterations
as meta-review insights feed back into evolution.

**Reality at `max_iterations=1`:** The framework runs:
Generation → Reflection → Ranking → Tournament → (repeat N times) → Meta-review → Evolution → Reflection → Ranking → Tournament → Proximity

The **meta-review** synthesises cross-cutting insights from all reviews. These insights
are passed to the evolution agent. But:
1. With `max_iterations=1`, the loop only runs once — there is no true iteration
2. The meta-review insights are passed as JSON context, which the evolution agent
   partially ignores (same problem as the supervisor)
3. The loop does not **regenerate** hypotheses based on portfolio gaps — it only evolves
   existing ones

**True iteration** would mean:
1. Meta-review identifies gaps (e.g., "no hypothesis addresses crop rotation")
2. Supervisor instructs generation agent: "Generate 2 new hypotheses specifically for
   the cultural pillar, covering crop rotation and trap crops"
3. New hypotheses enter the tournament and compete against existing ones

**Recommendation:** Implement **directed regeneration** — if meta-review identifies
uncovered IWM pillars, trigger a targeted generation call for those pillars.

---

### W7. Portfolio Breadth vs. Depth Tradeoff

**Observed portfolio with `N=5`:**
- 2 technology-heavy (UAV/VRT/CNN)
- 1 agrochemical (strigolactone + biocontrol)
- 1 physical/cultural (tillage + solarisation + biofumigation)
- 1 multi-pillar commercial (Clearfield + ethephon)

**Missing from the portfolio:**
- Crop rotation with non-hosts (simplest, cheapest, most farmer-accessible)
- Trap crops (linseed, flax)
- Resistant/tolerant tomato variety trials
- Manual removal at early emergence
- Irrigation management timing (reduce pre-transplant irrigation to limit germination cues)

The generation prompt heavily emphasises precision technology (UAV, VRT, ML), biasing
the portfolio toward capital-intensive high-tech solutions. This is appropriate for a
research institution with equipment access but misrepresents the full IWM solution space
for Mediterranean smallholders.

**Recommendation:** Add an explicit **portfolio balance constraint** to the supervisor:
*"At least 1 hypothesis must be implementable with no specialized equipment. At least 1
must focus on cultural/preventive management (rotation, variety selection). Technology
hypotheses should not exceed 40% of the portfolio."*

---

### W8. The Proximity/Clustering Step at Small Scale

With `N=5` hypotheses, the proximity analysis:
- Assigned 3 clusters (3 singletons, 2 ungrouped)
- Added 18s of API time and ~$0.005 of cost
- Provided no actionable output — with 5 hypotheses, a human can visually assess
  redundancy in seconds

**At N=20–50 hypotheses**, clustering becomes valuable. At N=5 it is overhead.

**Recommendation:** Make proximity analysis conditional on `N >= 10`.

---

## Part III — Summary Table

| Issue | Type | Severity | Status |
|-------|------|----------|--------|
| Claude content filter on IWM content | Technical + Model | High | Workaround: use gpt-4.1-mini |
| swarms message history in user message | Technical | High | Fixed: DirectLLMAgent |
| JSON output from agents | Technical | High | Fixed: hard instruction + JSON mode |
| GPT-5 not accessible with current key | Technical | Low | Open |
| GitHub push issues | Technical | Medium | Open (next session) |
| `save_state()` not implemented | Technical | Low | Open |
| Tournament generates weak Elo signal at N=5 | Conceptual | Medium | Recommendation: multi-criterion scoring |
| Supervisor redundant at small scale | Conceptual | Low | Recommendation: make conditional |
| Hypotheses are research protocols, not hypotheses | Conceptual | High | Recommendation: sharpening instruction |
| Evolution optimises for complexity, not parsimony | Conceptual | High | Recommendation: parsimony constraint |
| Score differentiation too narrow (0.72–0.87) | Conceptual | Medium | Recommendation: adversarial reviewer |
| Loop does not truly iterate | Conceptual | Medium | Recommendation: directed regeneration |
| Portfolio biased toward high-tech solutions | Conceptual | Medium | Recommendation: balance constraint in supervisor |
| Proximity analysis is overhead at N<10 | Conceptual | Low | Recommendation: conditional activation |

---

## Next Session Priorities

1. Fix GitHub push issue
2. Implement **hypothesis sharpening** in evolution prompt (parsimony constraint)
3. Add **adversarial review** agent or adversarial pass in reflection
4. Implement **directed regeneration** from meta-review gaps
5. Add **portfolio balance constraint** to supervisor prompt (crop rotation, trap crops mandatory)
6. Validate with `max_iterations=2`, `N=10` to test if true iteration improves quality
7. Consider replacing tournament winner/loser with multi-dimensional head-to-head scoring
