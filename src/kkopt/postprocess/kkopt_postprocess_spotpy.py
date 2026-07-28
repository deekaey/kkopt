# kkopt_postprocess.py
import os
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spotpy
from SALib.analyze import sobol, morris as morris_analyze


def temp( df_used, param_cols, project, method, like_type):
    # Prepare DataFrame with cleaned parameter names
    df_params = df_used[param_cols].copy()
    clean_names = [
        c[3:] if c.startswith("par") else c
        for c in param_cols
    ]
    df_params.columns = clean_names

    # Compute correlation matrix
    corr = df_params.corr(method="pearson")

    # Plot heatmap
    plt.figure(figsize=(0.5 * len(clean_names) + 4, 0.5 * len(clean_names) + 4))
    sns.heatmap(
        corr,
        xticklabels=clean_names,
        yticklabels=clean_names,
        cmap="coolwarm",
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
        square=True,
        cbar_kws={"label": "Pearson correlation"},
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.title("Parameter correlation (MCMC)")
    plt.tight_layout()

    suffix = _rep_suffix(project)
    corr_plot_path = os.path.join(
        project.output_dir,
        f"{project.setting.output}{suffix}_parameters_corr_{method.lower()}.png",
    )
    plt.savefig(corr_plot_path, dpi=300)
    plt.close()

