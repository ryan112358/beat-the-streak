import pandas as pd
import numpy as np
import scipy
from pandarallel import pandarallel

pandarallel.initialize()


def normal_log_losses(data, missing_mask):
    """Compute the per-example log loss under a single univariante gaussian model.

    Args:
        data: A 1- or 2-dimensional array of data.
        missing_mask: A 1- or 2-dimensional boolean array, same shape as data.

    Returns:
        A 1- or 2-dimensional array of log losses per data point under a gaussian model.
    """
    if data.ndim == 1:
        with np.errstate(divide="ignore"):
            mean = (data * missing_mask).sum() / missing_mask.sum()
            diff = data - mean
            variance = (diff**2 * missing_mask).sum() / missing_mask.sum()
            std = np.sqrt(variance)

            return -scipy.stats.norm.logpdf(data, loc=mean, scale=std)
    else:
        result = [normal_log_losses(x, y) for x, y in zip(data.T, missing_mask.T)]
        return np.vstack(result).T

def categorical_log_losses(data, missing_mask):
    if data.ndim == 1:
        minlength = data.max() + 2
        probas = np.bincount(data[missing_mask], minlength=minlength) / missing_mask.sum()
        return -np.nan_to_num(np.log(probas[data]))
    else:
        result = [categorical_log_losses(x, y) for x, y in zip(data.T, missing_mask.T)]
        return np.vstack(result).T


def independent_baseline(
    pitches: pd.DataFrame, 
    categorical_columns: list[str],
    numerical_columns: list[str], 
    groupby_cols: list = ["pitcher"]
):
    def fun(group):
        x = group[numerical_columns].values
        missing_mask = ~np.isnan(x)
        x = np.nan_to_num(x)
        result_num = normal_log_losses(x, missing_mask)

        x = group[categorical_columns].values
        missing_mask = x != -1

        result_cat = categorical_log_losses(x, missing_mask)
        return pd.DataFrame(
            data=np.hstack([result_cat, result_num]),
            columns=categorical_columns+numerical_columns, 
            index=group.index
        )

    losses = pitches.groupby(
        groupby_cols, observed=True, group_keys=False
    ).parallel_apply(fun).reindex(pitches.index)

    missing_mask_num = ~np.isnan(pitches[numerical_columns].values)
    missing_mask_cat = pitches[categorical_columns].values != -1
    missing_mask = np.hstack([missing_mask_cat, missing_mask_num])
    per_feature_losses = (losses * missing_mask).sum(axis=0) / missing_mask.sum(axis=0)

    return per_feature_losses
