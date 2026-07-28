
def _rep_suffix(project) -> str:
    """
    Return a suffix encoding the number of repetitions, e.g. '_N6000'.
    Falls back to empty string if repetitions is missing.
    """
    reps = getattr(project.setting, "repetitions", None)
    if reps is None:
        return ""
    try:
        n = int(reps)
    except Exception:
        return ""
    return f"_N{n}"
