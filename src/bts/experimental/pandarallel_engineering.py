"""Derive new features from raw data based on sliding window statistics."""
import numpy as np
import pandas as pd
import os
from IPython import embed
from bts.data.load import *
from glob import glob
import functools
import itertools
import time
from tqdm.auto import tqdm
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)
#tqdm.pandas(desc="Progress")

SINGLEARITY_COLUMNS_ALL = [
    "woba",
    "field_out",
    "strikeout",
    "single",
    "walk",
    "double",
    "home_run",
    "force_out",
    "grounded_into_double_play",
    "hit_by_pitch",
    "field_error",
    "sac_fly",
    "sac_bunt",
    "triple",
    "intent_walk",
    "double_play",
    "hit",
    "ehit",
    "estimated_woba_using_speedangle",
    "estimated_ba_using_speedangle",
]
SINGLEARITY_COLUMNS_SHORT = [
    "pf_hit_1095",
    "single",
    "double",
    "triple",
    "home_run",
    "walk",
    "strikeout",
    "woba",
    "hit",
    "ehit",
    "estimated_woba_using_speedangle",
    "estimated_ba_using_speedangle",
]
# Most common pitch types
PITCH_TYPES = ["FF", "SI", "SL", "CH", "CU", "FC", "KC", "FA", "FS"]
# Most common descriptions.  TODO: handle other descriptions
DESCRIPTIONS = ["ball", "hit_into_play", "foul", "called_strike", "swinging_strike"]
ZONES = [str(x) for x in range(1, 15) if x != 10]


def compute_park_factors(atbats):
    # Some test set leakage occurs here, ignoring for now
    # TODO: compute the 3year moving average park factor
    stuff = atbats.copy()
    columns = ["single", "double", "triple", "home_run"]
    for e in columns:
        stuff[e] = atbats.events == e
    columns += ["hit"]
    batters = stuff.groupby(["home", "batter_team"])[columns].sum()
    pitchers = stuff.groupby(["home", "pitcher_team"])[columns].sum()
    games = stuff.groupby(["home", "batter_team"]).game_pk.nunique()
    # return batters, pitchers, games
    numerator = (batters.loc[True] + pitchers.loc[False]).div(games.loc[True], axis=0)
    denominator = (batters.loc[False] + pitchers.loc[True]).div(
        games.loc[False], axis=0
    )
    return (numerator / denominator).rename(columns=lambda s: "pf_%s" % s).fillna(1.0)


def augment_park_factors(atbats):
    pfs = compute_park_factors(atbats)
    pfs.index.name = "ballpark"
    aug = pfs.loc[atbats.ballpark]
    aug.index = atbats.index
    return pd.concat([atbats, aug], axis=1)


def sliding_mean(df: pd.DataFrame, days: int = 365, thresh: int = 40):
    """Compute a sliding mean over the given number of days.

    Notes: 
        1. All columns of df other than game_date must be numeric (float, int, bool).

    :param df: the source data frame, must have game_date column.
    :param days: The number of days in the sliding window.
    :param thresh: The minimum number of observations in the time window.

    :returns: A new dataframe, with one entry per unique game_date, where 
        each entry is the rolling mean of previous `days` observations or NaN.
    """
    # closed=left ensures no test-set leakage.  i.e., the feature for a given game_date
    # only depends on data from before that game_date.
    roll = df.rolling("%dd" % days, on="game_date", closed="left", min_periods=thresh)
    # Grab the first record for each game_date
    return roll.mean().groupby("game_date").head(n=1)


def exponential_moving_avg(df: pd.DataFrame, default: pd.Series, alpha: float = 0.01):
    """Compute an exponential moving average with the given default value.

    Notes: 
        1. All columns of df other than game_date must be numeric (float, int, bool).
        2. NaNs are automatically handled, when a NaN observation is observed,
            the moving average is backfilled with most recent value.
        3. The game_date of previous observations does not affect this calculation, only the
            relative position in the data frame.

    :param df: the soruce data frame, must have game_date column.
    :param default: The default value to use before any data has been observed.
        Should be a series with one entry per column of df (other than game_date).
    :param alpha: The decay parameter.  Lower values cause larger recent data
        to have more impact on the moving average.

    :return: A new dataframe, with one entry per unique game_date, where
        each entry is the exponential moving average of all previous observations.
    """
    default = default.to_frame().T
    # compute moving average and shift so that feature only depends on past data.
    ewm = (
        pd.concat([default, df.drop(columns="game_date")])
        .ewm(alpha=alpha, adjust=False)
        .mean()
        .shift(1)
    ).iloc[1:]
    # Take the first item for each game_date.
    ewm.insert(0, "game_date", df["game_date"])
    return ewm.groupby("game_date").head(n=1)


def ewm_by_partition(
    df: pd.DataFrame,
    groupby_keys: list[str],
    columns: list[str],
    alpha: float = 0.01,
    default: pd.Series | None = None,
):
    """Compute the exponential moving average broken down by group for each column.

    :param df: The source data frame, must have 'game_date' column.
    :param groupby_keys: A list of categorical columns in df to group by.
    :param columns: A list of numeric columns to compute moving averages for.
    :param alpha: The decay parameter.  Lower values cause larger recent data
        to have more impact on the moving average.
    
    :returns: A new dataframe, with one entry per unique (groupby_key, game_date).
    """
    if default is None:
        default = df[columns].mean()
    # group data and fetch relevant numeric and game_date columns
    groups = df.groupby(groupby_keys, observed=True)[["game_date"] + columns]
    # Apply exponential moving avg to each group
    apply_fn = functools.partial(exponential_moving_avg, default=default, alpha=alpha)
    result = groups.parallel_apply(apply_fn).reset_index(groupby_keys).sort_index()
    rename_fn = (
        lambda s: "_".join(groupby_keys + [s, str(alpha)])
        if not s in ["game_date"] + groupby_keys
        else s
    )
    return result.rename(columns=rename_fn).reset_index(drop=True)


def null_preserving_dummies(series):
    dummies = pd.get_dummies(series)
    mask = series.notnull().replace({False: None})
    return dummies.multiply(mask, axis=0)


def conditional_dummies(
    df: pd.DataFrame, dummy_cols: list[str], dummy_values: list[list]
):
    """Compute conditional dummies.
 
    Note:
        1. If len(dummy_cols) = 1, this function simply computes ordinary dummies
            where all entries of the resulting dataframe are True/False.
        1. If len(dummy_cols) > 1, the order DOES matter.  We will compute the moving
            average of the first dummy column conditioned on other columns.  In general,
            there will be NaNs corresponding to entries of the dummy encoding corresponding
            to conditioned variables that were not observed.

    :param df: The source dataframe
    :param dummy_cols: A list of categorical columns whose conditional dummy encoding is
        desired.
    :param dummy_values: A list of lists, where each list includes possible values for
        the corresponding dummy_col.

    :returns: A new dataframe with the same number of rows as df, where the dummy_cols
        have been converted into a conditional one hot encoding.
    """
    dummies = [
        null_preserving_dummies(df[col])[vals]
        for col, vals in zip(dummy_cols, dummy_values)
    ]
    dummies = pd.concat(dummies, axis=1)
    if len(dummy_cols) == 1:
        return dummies  # optimization to avoid unnecessary work

    # Refactoring based on performance warning
    fragmented_df = {}
    for values in itertools.product(*dummy_values):
        col = "_".join(map(str, values))
        # Don't replace false with None here
        # Copy needed due to *= below
        fragmented_df[col] = dummies[values[0]].copy()
        for c in values[1:]:
            # We don't want to update the moving average for this dummy category
            # since it wasn't observed.
            fragmented_df[col] *= dummies[c].replace({False: None})

    # TODO: think carefully about what the desired behavior is here wrt:
    # (1) true nulls and (2) spurious nulls from excluded categories.
    # null_preserving_dummies makes it so nulls in any dummy_col will not be
    # affect the default mean or the moving averages.
    # current behavior is (I believe)
    # - true nulls: do not affect the rolling averages (backfilled appropriately)
    # - spurious nulls (those not appearing in dummy_values): does affect rolling avgs
    #   and default initialization.
    return pd.DataFrame(fragmented_df)


# TODO: create similar helper function for sliding_mean_by_partition
# TODO: think about if this logic can be moved to exponential_moving_avg
#       so that it uses less memory.
# Update: Seems like it can, but passing in default value will have to change.
def ewm_dummies_by_partition(
    df: pd.DataFrame,
    groupby_keys: list[str],
    dummy_cols: list[str],
    dummy_values=list[list],
    alpha: float = 0.01,
):
    """Exponential moving average of dummy columns.

    Note:
        1. If len(dummy_cols) > 1, the order DOES matter.  We will compute the moving
            average of the first dummy column conditioned on other columns.

    :param df: The source data frame, must have 'game_date' column.
    :param groupby_keys: A list of categorical columns in df to group by.
    :param dummy_cols: A list of categorical columns to compute moving averages for. 
    :param dummy_values: A list of values for each dummy column that we want
        moving averages for.
    :param alpha: Moving average decay parameter.

    :returns: A new dataframe, with one entry per unique (groupby_key, game_date).
    """
    derived_df = conditional_dummies(df, dummy_cols, dummy_values)
    numeric_columns = list(derived_df.columns)
    default = derived_df.mean()
    derived_df["game_date"] = df["game_date"]
    derived_df[groupby_keys] = df[groupby_keys]
    return ewm_by_partition(
        derived_df, groupby_keys, numeric_columns, alpha=alpha, default=default
    )


def sliding_mean_by_partition(
    df, groupby_keys, columns, days=365, thresh=40, impute=True
):
    """Compute sliding averages across columns borken down by groupby_keys.

    :param df: An input dataframe with a 'game_date' column
    :param groupby_keys: A list of categorical columns to break down sliding means.
    :param columns: A list of numeric columns to take sliding means of.
    :param days: The length of time in days to compute sliding means.
    :param thresh: The minimum number of records required in the window.

    :returns: A dataframe with derived features that can be joined with 
        other DataFrames using join_key = ['game_date'] + groupby_keys
    """
    # Group data, fetch relevant numeric and game_date columns
    groups = df.groupby(groupby_keys, observed=True)[["game_date"] + columns]
    # Apply sliding window to the entire dataset
    apply_fn = lambda df: sliding_mean(df, days, thresh)
    average = apply_fn(df[["game_date"] + columns]).bfill()  # fillna(method="bfill")
    # Apply sliding window mean to each group
    per_partition = (
        groups.parallel_apply(apply_fn)
        .reset_index(level=groupby_keys)
        .reset_index(drop=True)
    )

    if impute:  # Missing value imputation.  Update NaNs with imputed values.
        per_partition["imputed"] = ~per_partition[columns[0]].notnull()
        imputed_vals = per_partition.drop(columns=columns).merge(
            average, how="left", on="game_date"
        )
        assert imputed_vals.shape[0] == per_partition.shape[0]
        per_partition.update(imputed_vals, overwrite=False)

    # Rename columns so that they are unique.
    rename_fn = (
        lambda s: "_".join(groupby_keys + [s, str(days)])
        if not s in ["game_date"] + groupby_keys
        else s
    )
    return per_partition.rename(columns=rename_fn)


def park_factor_3year(atbats):
    days = 365 * 3
    events = ["single", "double", "triple", "home_run", "hit", "woba"]
    home_hits = sliding_mean_by_partition(atbats, ["home_team"], events, days, 40)
    away_hits = sliding_mean_by_partition(atbats, ["away_team"], events, days, 40)
    home_hits = home_hits.rename(columns={"home_team": "ballpark"}).sort_values(
        "game_date"
    )
    away_hits = away_hits.rename(columns={"away_team": "ballpark"}).sort_values(
        "game_date"
    )
    tmp = pd.merge_asof(home_hits, away_hits, on="game_date", by="ballpark")
    for e in events:
        col1 = "home_team_%s_%d" % (e, days)
        col2 = "away_team_%s_%d" % (e, days)
        col = "pf_%s_%d" % (e, days)
        tmp[col] = tmp.pop(col1) / tmp.pop(col2)
    return tmp


def batter_3year(atbats):
    return sliding_mean_by_partition(
        atbats, ["batter"], SINGLEARITY_COLUMNS_ALL, 365 * 3, 40
    )


def pitcher_3year(atbats):
    return sliding_mean_by_partition(
        atbats, ["pitcher"], SINGLEARITY_COLUMNS_ALL, 365 * 3, 40
    )


def batter_1year(atbats):
    return sliding_mean_by_partition(
        atbats, ["batter"], SINGLEARITY_COLUMNS_ALL, 365, 40
    )


def pitcher_1year(atbats):
    return sliding_mean_by_partition(
        atbats, ["pitcher"], SINGLEARITY_COLUMNS_ALL, 365, 40
    )


def batter_recent(atbats):
    return sliding_mean_by_partition(
        atbats, ["batter"], SINGLEARITY_COLUMNS_SHORT, 21, 20
    )


def pitcher_recent(atbats):
    return sliding_mean_by_partition(
        atbats, ["pitcher"], SINGLEARITY_COLUMNS_SHORT, 21, 20
    )


def batter_vs_pitcher_3year(atbats):
    return sliding_mean_by_partition(
        atbats, ["batter", "pitcher"], SINGLEARITY_COLUMNS_SHORT, 365 * 3, 10
    )


def pitcher_type(pitches, alpha=0.01):
    return ewm_dummies_by_partition(
        df=pitches,
        groupby_keys=["pitcher"],
        dummy_cols=["pitch_type"],
        dummy_values=[PITCH_TYPES],
        alpha=alpha,
    )


def batter_type(pitches, alpha=0.01):
    return ewm_dummies_by_partition(
        df=pitches,
        groupby_keys=["batter"],
        dummy_cols=["pitch_type"],
        dummy_values=[PITCH_TYPES],
        alpha=alpha,
    )


def pitcher_description(pitches, alpha=0.01):
    return ewm_dummies_by_partition(
        df=pitches,
        groupby_keys=["pitcher"],
        dummy_cols=["description"],
        dummy_values=[DESCRIPTIONS],
        alpha=alpha,
    )


def pitcher_zone(pitches, alpha=0.01):
    return ewm_dummies_by_partition(
        df=pitches,
        groupby_keys=["pitcher"],
        dummy_cols=["zone"],
        dummy_values=[ZONES],
        alpha=alpha,
    )


def pitcher_zone_outcome(pitches, alpha=0.01):
    return ewm_dummies_by_partition(
        df=pitches,
        groupby_keys=["pitcher"],
        dummy_cols=["description", "zone"],
        dummy_values=[DESCRIPTIONS, ZONES],
        alpha=alpha,
    )


def batter_zone(pitches, alpha=0.01):
    return ewm_dummies_by_partition(
        df=pitches,
        groupby_keys=["batter"],
        dummy_cols=["zone"],
        dummy_values=[ZONES],
        alpha=alpha,
    )


def batter_outcome(pitches, alpha=0.01):
    return ewm_dummies_by_partition(
        df=pitches,
        groupby_keys=["batter"],
        dummy_cols=["description"],
        dummy_values=[DESCRIPTIONS],
        alpha=alpha,
    )


def pitcher_outcome(pitches, alpha=0.01):
    return ewm_dummies_by_partition(
        df=pitches,
        groupby_keys=["pitcher"],
        dummy_cols=["description"],
        dummy_values=[DESCRIPTIONS],
        alpha=alpha,
    )


def batter_zone_outcome(pitches, alpha=0.01):
    return ewm_dummies_by_partition(
        df=pitches,
        groupby_keys=["batter"],
        dummy_cols=["description", "zone"],
        dummy_values=[DESCRIPTIONS, ZONES],
        alpha=alpha,
    )


def pitcher_type_outcome(pitches, alpha=0.01):
    return ewm_dummies_by_partition(
        df=pitches,
        groupby_keys=["pitcher"],
        dummy_cols=["description", "pitch_type"],
        dummy_values=[DESCRIPTIONS, PITCH_TYPES],
        alpha=alpha,
    )


def all_pitcher_ewm_features(pitches, alpha):
    return [
        pitcher_type_outcome(pitches, alpha),
        pitcher_type(pitches, alpha),
        pitcher_zone(pitches, alpha),
        pitcher_zone_outcome(pitches, alpha),
        pitcher_outcome(pitches, alpha),
    ]


def all_batter_ewm_features(pitches, alpha):
    return [
        batter_type_outcome(pitches, alpha),
        batter_type(pitches, alpha),
        batter_zone(pitches, alpha),
        batter_zone_outcome(pitches, alpha),
        batter_outcome(pitches, alpha),
    ]


def batter_type_outcome(pitches, alpha=0.01):
    return ewm_dummies_by_partition(
        df=pitches,
        groupby_keys=["batter"],
        dummy_cols=["description", "pitch_type"],
        dummy_values=[DESCRIPTIONS, PITCH_TYPES],
        alpha=alpha,
    )


def merge_features(features, key):
    df = features[0].sort_values("game_date")
    for feature in features[1:]:
        df = pd.merge(df, feature, on=key, how="left")
        # feature = feature.sort_values('game_date')
        # df = pd.merge_asof(df, feature, on='game_date', by=key[:-1])
    return df


def all_batter_features(atbats, pitches):
    features = (
        [batter_3year(atbats), batter_1year(atbats), batter_recent(atbats)]
        + all_batter_ewm_features(pitches, 1 / 256)
        + all_batter_ewm_features(pitches, 1 / 1024)
        + all_batter_ewm_features(pitches, 1 / 8192)
        + [batter_7day_woba(atbats)]
    )
    return merge_features(features, ["batter", "game_date"])


def all_pitcher_features(atbats, pitches):
    features = (
        [pitcher_3year(atbats), pitcher_1year(atbats), pitcher_recent(atbats)]
        + all_pitcher_ewm_features(pitches, 1 / 256)
        + all_pitcher_ewm_features(pitches, 1 / 1024)
        + all_pitcher_ewm_features(pitches, 1 / 8192)
    )
    return merge_features(features, ["pitcher", "game_date"])


def all_park_features(atbats):
    features = [park_factor_3year(atbats)]
    return merge_features(features, ["ballpark", "game_date"])


def all_batter_team_features(atbats):
    # TODO: Move this logic to a helper function
    pa_per_game = (
        atbats.groupby(["batter_team", "game_pk", "game_date"], observed=True)
        .hit.count()
        .rename("PA")
        .reset_index()
        .sort_values("game_date")
    )
    pa = sliding_mean_by_partition(pa_per_game, ["batter_team"], ["PA"], 60, 10)
    features = [pa]
    return merge_features(features, ["batter_team", "game_date"])


def batter_7day_woba(atbats):
    return sliding_mean_by_partition(
        atbats,
        ["batter"],
        ["estimated_woba_using_speedangle"],
        days=7,
        thresh=20,
        impute=False,
    )


def batter_pthrows_game_year_features(atbats):
    return sliding_mean_by_partition(
        atbats,
        ["batter", "game_year", "p_throws"],
        ["estimated_ba_using_speedangle"],
        days=365,
        thresh=20,
        impute=False,
    )


def batter_game_year_features(data):
    L28GWAH = sliding_mean_by_partition(
        data,
        ['batter', 'game_year'],
        ['hit_at_least_one', 'ehit_at_least_one'],
        days=28,
        thresh=10,
        impute=False
    )

    FSGWAH = sliding_mean_by_partition(
        data,
        ['batter', 'game_year'],
        ['hit_at_least_one', 'ehit_at_least_one'],
        days=365,
        thresh=10,
        impute=False
    )

    return merge_features([L28GWAH, FSGWAH], ['batter', 'game_year', 'game_date'])

if __name__ == "__main__":
    t0 = time.time()
    data = load_data()
    atbats = load_atbats()
    pitches = load_pitches()
    print("Loaded Data", time.time() - t0)
    # Removing nulls makes the merge between pitch-features and atbat-features to get
    # a little messy.  Commenting out this line is the fix for now, although something
    # different could be done in the future.
    # pitches = pitches[pitches.pitch_type.notnull()]
    """
    dummy1 = pd.get_dummies(pitches.pitch_type)[PITCH_TYPES]
    dummy2 = pd.get_dummies(pitches.description)[DESCRIPTIONS]
    dummy3 = pd.get_dummies(pitches.zone.astype("str"))[ZONES]
    pitches = pd.concat([pitches, dummy1, dummy2], axis=1)
    for col1 in PITCH_TYPES:
        for col2 in DESCRIPTIONS:
            pitches[col1 + "_" + col2] = (
                pitches[col1].replace({False: None}) * pitches[col2]
            )
    """
    dummy = pd.get_dummies(atbats["events"])
    atbats = pd.concat([atbats, dummy], axis=1)
    weights = dict(
        walk=0.69,
        hit_by_pitch=0.719,
        single=0.87,
        double=1.217,
        triple=1.529,
        home_run=1.94,
    )
    atbats["woba"] = sum(weights[e] * atbats[e] for e in weights)
    print("Preprocessed Data", time.time() - t0)

    park_features = all_park_features(atbats)
    atbats = atbats.merge(park_features, how="left", on=["ballpark", "game_date"])
    print("Computed and Merged Park Features", time.time() - t0)

    batter_features = all_batter_features(atbats, pitches)
    print("Computed Batter Features", time.time() - t0)

    pitcher_features = all_pitcher_features(atbats, pitches)
    print("Computed Pitcher Features", time.time() - t0)

    batter_team_features = all_batter_team_features(atbats)
    print("Computed Team Features", time.time() - t0)

    ironcond0r_batter_game_year_features = batter_game_year_features(data)
    ironcond0r_batter_pthrows_game_year_features = batter_pthrows_game_year_features(atbats)

    # Would be ideal to not have any nulls here
    # Temporary solution is to throw out data for first 3 years
    # for F in [park_features, batter_features, pitcher_features, batter_team_features]:
    #    assert F.notnull().all().all()

    # embed()

    ROOT = os.environ["BTSDATA"]
    park_features.to_parquet(
        f"{ROOT}/ballpark_features.parquet.gzip", compression="gzip"
    )
    batter_features.to_parquet(
        f"{ROOT}/batter_features.parquet.gzip", compression="gzip"
    )
    pitcher_features.to_parquet(
        f"{ROOT}/pitcher_features.parquet.gzip", compression="gzip"
    )
    batter_team_features.to_parquet(
        f"{ROOT}/batter_team_features.parquet.gzip", compression="gzip"
    )
    ironcond0r_batter_pthrows_game_year_features.to_parquet(
        f"{ROOT}/batter_pthrows_game_year_features.parquet.gzip", compression="gzip"
    )
    ironcond0r_batter_game_year_features.to_parquet(
        f"{ROOT}/batter_game_year_features.parquet.gzip", compression="gzip"
    )

    # engineered_data = engineered_data[engineered_data.game_date >= '2013-01-01']
    # num_data = engineered_data.filter(regex='1095|365|21|60')
