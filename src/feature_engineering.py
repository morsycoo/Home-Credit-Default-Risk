import pandas as pd


def merge_features(base_df, feature_df):
    """
    Merge aggregated features using SK_ID_CURR.
    """

    return base_df.merge(
        feature_df,
        on="SK_ID_CURR",
        how="left"
    )