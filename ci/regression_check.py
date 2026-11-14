# ci/regression_check.py
import sys

import pandas as pd

from config.settings import REGRESSION_TOLERANCE
from storage.supabase_client import client, is_supabase_configured


def run_regression_check():
    print("🔍 RUNNING AUTOMATED LLM QUALITY REGRESSION CHECK...")
    print("=" * 65)

    # CI runs may not have database credentials (for example, on external PRs).
    # Avoid dereferencing a missing client; report the skipped gate explicitly.
    if not is_supabase_configured() or client is None:
        print("⚠️ Supabase is not configured; skipping the regression comparison.")
        print("   Set SUPABASE_URL and SUPABASE_KEY in the repository or environment secrets to enable it.")
        return 0

    try:
        response = client.table("eval_results").select("*").execute()
    except Exception as exc:
        print(f"❌ Could not fetch evaluation history from Supabase: {exc}", file=sys.stderr)
        return 1

    data = response.data
    if not data or len(data) < 2:
        print("⚠️ Not enough evaluation results in Supabase to perform a regression comparison.")
        print("✅ Passing check as initial baseline setup.")
        return 0

    df = pd.DataFrame(data)
    required_columns = {"id", "eval_version", "judge_faithfulness", "bertscore_f1"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        print(
            "❌ Evaluation results are missing required columns: "
            + ", ".join(sorted(missing_columns)),
            file=sys.stderr,
        )
        return 1

    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df = df.dropna(subset=["id"]).sort_values("id")
    versions = df["eval_version"].dropna().unique()

    if len(versions) < 2:
        print("ℹ️ Only one evaluation version found. Setting it as the baseline.")
        return 0

    baseline_version, current_version = versions[-2], versions[-1]
    print(f"📊 Comparing Current Run [{current_version}] vs Baseline Run [{baseline_version}]")

    baseline_df = df[df["eval_version"] == baseline_version]
    current_df = df[df["eval_version"] == current_version]
    baseline_faith = baseline_df["judge_faithfulness"].mean()
    current_faith = current_df["judge_faithfulness"].mean()
    baseline_bert = baseline_df["bertscore_f1"].mean()
    current_bert = current_df["bertscore_f1"].mean()

    print("\n--- BENCHMARK COMPARISON SUMMARY ---")
    print(f"Faithfulness : Baseline = {baseline_faith:.2f} | Current = {current_faith:.2f}")
    print(f"BERTScore    : Baseline = {baseline_bert:.3f} | Current = {current_bert:.3f}")

    has_regression = False
    reasons = []

    if pd.notna(baseline_faith) and baseline_faith > 0:
        faith_change = (current_faith - baseline_faith) / baseline_faith
        if faith_change < -REGRESSION_TOLERANCE:
            has_regression = True
            reasons.append(
                f"Faithfulness dropped by {abs(faith_change) * 100:.1f}% "
                f"(Threshold: {REGRESSION_TOLERANCE * 100}%)"
            )

    if pd.notna(baseline_bert) and baseline_bert > 0:
        bert_change = (current_bert - baseline_bert) / baseline_bert
        if bert_change < -REGRESSION_TOLERANCE:
            has_regression = True
            reasons.append(
                f"BERTScore dropped by {abs(bert_change) * 100:.1f}% "
                f"(Threshold: {REGRESSION_TOLERANCE * 100}%)"
            )

    print("=" * 65)
    if has_regression:
        print("❌ QUALITY REGRESSION DETECTED! BUILD FAILED.")
        for reason in reasons:
            print(f"   • {reason}")
        print("=" * 65)
        return 1

    print("✅ QUALITY CHECKS PASSED. NO SIGNIFICANT REGRESSION DETECTED.")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(run_regression_check())
