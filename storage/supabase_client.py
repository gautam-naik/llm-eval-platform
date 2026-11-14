# storage/supabase_client.py
from supabase import create_client
from config.settings import SUPABASE_URL, SUPABASE_KEY


def is_supabase_configured() -> bool:
    return bool(SUPABASE_URL and SUPABASE_KEY)


def get_client():
    if not is_supabase_configured():
        return None
    return create_client(SUPABASE_URL, SUPABASE_KEY)


client = get_client()


def _require_client():
    if client is None:
        raise RuntimeError("Supabase is not configured. Set SUPABASE_URL and SUPABASE_KEY.")
    return client


def insert_test_case(prompt, ref_answer, context, category, attack_type="none"):
    supabase = _require_client()
    data = {
        "prompt": prompt,
        "reference_answer": ref_answer,
        "source_context": context,
        "category": category,
        "attack_type": attack_type,
        "version": "v2.0"  # New version tag for synthetic data
    }
    supabase.table("test_cases").insert(data).execute()


def get_test_cases(version=None):
    if client is None:
        return []

    query = client.table("test_cases").select("*")
    if version:
        query = query.eq("version", version)
    response = query.execute()
    return response.data


def insert_eval_result(result):
    if client is None:
        return None
    client.table("eval_results").insert(result).execute()


def get_eval_results(version=None):
    if client is None:
        return []

    query = client.table("eval_results").select("*")
    if version:
        query = query.eq("eval_version", version)
    response = query.execute()
    return response.data
