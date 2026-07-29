# kkopt_postprocess.py
import os
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spotpy
from SALib.analyze import sobol, morris as morris_analyze

from kkopt.postprocess.kkopt_postprocess_spotpy import spotpy_postprocess
from kkopt.postprocess.common import _rep_suffix

def postprocess(project):
    method = getattr(project.setting, "method", "").lower()

    if method in ["mcmc", "fast", "lhs"]:
        spotpy_postprocess( project, method=method)

    elif method == "sobol":
        # if indices don't exist yet, compute them from Y
        suffix = _rep_suffix(project)
        base = project.setting.output + "_sobol" + suffix
        S1_file = base + "_S1.csv"
        ST_file = base + "_ST.csv"
        if not (os.path.exists(S1_file) and os.path.exists(ST_file)):
            salib_sobol_analysis_from_y(project)
        salib_sobol_postprocess(project)
    elif method == "morris":
        suffix = _rep_suffix(project)
        base = project.setting.output + "_morris" + suffix
        indices_file = base + "_indices.csv"
        #if not os.path.exists(indices_file):
        salib_morris_analysis_from_y(project)
        salib_morris_postprocess(project)
    else:
        print(f"[postprocess] No postprocessing implemented for method='{method}'")
