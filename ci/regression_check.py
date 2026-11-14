# ci/regression_check.py
import sys
import pandas as pd
from storage.supabase_client import client, is_supabase_configured
from config.settings import REGRESSION_TOLERANCE


def run_regression_check():
    print("🔍 RUNNING AUTOMATED LLM QUALITY REGRESSION CHECK...")
    print("=" * 65)

    # GitHub does not expose repository secrets to some pull_request workflows.
    # In that case the Supabase client is intentionally unavailable, so skip
    # the database-backed gate rather than crashing on client.table(...).
    if not is_supabase_configured() or client is None:
        print("⚠️ Supabase credentials are unavailable; skipping the regression comparison.")
        print("   Configure SUPABASE_URL and SUPABASE_KEY in GitHub Actions secrets to enable it.")
        return 0

    try:
        res = client.table("eval_results").select("*").execute()
    except Exception as e:
        print(f"❌ Failed to fetch evaluation results from Supabase: {e}", file=sys.stderr)
        return 1

    data = res.data
    if not data or len(data) < 2:
        print("⚠️ Not enough evaluation runs in Supabase to perform a regression comparison.")
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
    b_faith = baseline_df["judge_faithfulness"].mean()
    c_faith = current_df["judge_faithfulness"].mean()
    b_bert = baseline_df["bertscore_f1"].mean()
    c_bert = current_df["bertscore_f1"].mean()

    print("\n--- BENCHMARK COMPARISON SUMMARY ---")
    print(f"Faithfulness : Baseline = {b_faith:.2f} | Current = {c_faith:.2f}")
    print(f"BERTScore    : Baseline = {b_bert:.3f} | Current = {c_bert:.3f}")

    reasons = []
    if pd.notna(b_faith) and b_faith > 0 and pd.notna(c_faith):
        faith_change = (c_faith - b_faith) / b_faith
        if faith_change < -REGRESSION_TOLERANCE:
            reasons.append(
                f"Faithfulness dropped by {abs(faith_change) * 100:.1f}% "
                f"(Threshold: {REGRESSION_TOLERANCE * 100}%)"
            )

    if pd.notna(b_bert) and b_bert > 0 and pd.notna(c_bert):
        bert_change = (c_bert - b_bert) / b_bert
        if bert_change < -REGRESSION_TOLERANCE:
            reasons.append(
                f"BERTScore dropped by {abs(bert_change) * 100:.1f}% "
                f"(Threshold: {REGRESSION_TOLERANCE * 100}%)"
            )

    print("=" * 65)
    if reasons:
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
