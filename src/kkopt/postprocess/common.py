
import numpy as np

def _rep_suffix( project) -> str:
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


def indent( elem, level=0):
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for e in elem:
            indent(e, level + 1)
            if not e.tail or not e.tail.strip():
                e.tail = i + "  "
        if not e.tail or not e.tail.strip():
            e.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def rmse( a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if b.ndim == 1:
        b = b[None, :]
    a_b, b_b = np.broadcast_arrays(a, b)
    return np.sqrt(np.nanmean((a_b - b_b) ** 2, axis=-1))


def rrmse( a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if b.ndim == 1:
        b = b[None, :]
    a_b, b_b = np.broadcast_arrays(a, b)
    denom = np.nanmean(np.abs(b_b), axis=-1)
    denom = np.where(denom == 0, np.nan, denom)
    return rmse(a_b, b_b) / denom


def r2( a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if b.ndim == 1:
        b = b[None, :]
    a_b, b_b = np.broadcast_arrays(a, b)
    b_mean = np.nanmean(b_b, axis=-1, keepdims=True)
    ss_tot = np.nansum((b_b - b_mean) ** 2, axis=-1)
    ss_res = np.nansum((a_b - b_b) ** 2, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 - (ss_res / ss_tot)
