import json

from ai_coscientist import AIScientistFramework

ai_coscientist = AIScientistFramework(
    model_name="gpt-4.1-mini",
    max_iterations=1,
    verbose=False,
    hypotheses_per_generation=5,
    tournament_size=4,
    evolution_top_k=2,
)

# Broomrape research goal
research_goal = (
    "Identify effective integrated management strategies"
    " for Phelipanche ramosa in processing tomato"
    " (Solanum lycopersicum) under Mediterranean"
    " irrigated conditions, using germination stimulants,"
    " biocontrol agents, and precision herbicide"
    " treatments to cut the soil seed bank by 60%"
    " over three seasons."
)

results = ai_coscientist.run_research_workflow(research_goal)

# Output results
print("\n--- IWM Research Workflow Results ---")
if "error" in results:
    print(f"Error: {results['error']}")
else:
    print("\n--- Top Ranked Hypotheses ---")
    for i, hy in enumerate(results["top_ranked_hypotheses"], 1):
        print(f"\n{'='*60}")
        print(f"Hypothesis {i}")
        print(f"{'='*60}")
        print(f"Text: {hy['text']}")
        print(f"Elo Rating: {hy['elo_rating']}")
        print(f"Score: {hy['score']:.2f}")
        print(
            f"Win Rate: {hy['win_rate']}%"
            f" ({hy['total_matches']} matches)"
        )
        if hy["reviews"]:
            last_review = hy["reviews"][-1]
            scores = last_review.get("scores", {})
            print("  Review Scores:")
            for k, v in scores.items():
                label = k.replace("_", " ").title()
                print(f"    {label}: {v}/5")

    print("\n--- Meta-Review Insights ---")
    summary = results["meta_review_insights"].get(
        "meta_review_summary",
        "No summary available.",
    )
    print(summary[:500] + "..." if len(summary) > 500 else summary)

    print("\n--- Execution Metrics ---")
    print(json.dumps(results["execution_metrics"], indent=2))
    print(f"\nTotal Time:" f" {results['total_workflow_time']:.2f}s")

try:
    ai_coscientist.save_state()
except Exception as e:
    print(f"Error saving state: {e}")
