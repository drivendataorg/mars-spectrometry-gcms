import pandas as pd 
import numpy as np 
from tqdm import tqdm 
from joblib import Parallel, delayed
import gc 
from sklearn.preprocessing import minmax_scale
import itertools
import os 

def drop_frac_and_He(df):
    """
    Rounds fractional m/z values, drops m/z values > 350, and drops carrier gas m/z

    Args:
        df: a dataframe representing a single sample, containing m/z values

    Returns:
        The dataframe without fractional m/z and carrier gas m/z
    """

    # rounds m/z fractional values
    df["rounded_mass"] = df["mass"].transform(round)

    # aggregates across rounded values
    df = df.groupby(["time", "rounded_mass"])["intensity"].aggregate("mean").reset_index()

    # drop m/z values greater than 350
    df = df[df["rounded_mass"] <= 500]

    # drop carrier gas
    df = df[df["rounded_mass"] != 4]

    return df


def remove_background_intensity(df):
    """
    Subtracts minimum abundance value

    Args:
        df: dataframe with 'mass' and 'intensity' columns

    Returns:
        dataframe with minimum abundance subtracted for all observations
    """

    df["intensity_minsub"] = df.groupby(["rounded_mass"])["intensity"].transform(
        lambda x: (x - x.min())
    )

    return df

def scale_intensity(df):
    """
    Scale abundance from 0-1 according to the min and max values across entire sample

    Args:
        df: dataframe containing abundances and m/z

    Returns:
        dataframe with additional column of scaled abundances
    """

    df["intensity_minsub"] = minmax_scale(df["intensity_minsub"].astype(float))

    return df

# Preprocess function
def preprocess_sample(df):
    df = drop_frac_and_He(df)
    df = remove_background_intensity(df)
    # df = scale_intensity(df)
    return df

timerange = pd.interval_range(start=0, end=50, freq=0.05)

# Make dataframe with rows that are combinations of all temperature bins and all m/z values
allcombs = list(itertools.product(timerange, [*range(0, 500)]))

allcombs_df = pd.DataFrame(allcombs, columns=["time_bin", "rounded_mass"])
print(allcombs_df.head())

def int_per_timebin(df):

    """
    Transforms dataset to take the preprocessed max abundance for each
    time range for each m/z value

    Args:
        df: dataframe to transform

    Returns:
        transformed dataframe
    """

    # Bin times
    df["time_bin"] = pd.cut(df["time"], bins=timerange)

    # Combine with a list of all time bin-m/z value combinations
    df = pd.merge(allcombs_df, df, on=["time_bin", "rounded_mass"], how="left")

    # Aggregate to time bin level to find max
    df = df.groupby(["time_bin", "rounded_mass"]).max("intensity_minsub").reset_index()
    # print('1',df.shape)
    # print(df.head())
    # Fill in 0 for intensity values without information
    df = df.replace(np.nan, 0)

    features = np.zeros((500, 999)) #seq_size x mass
    for t in range(500):
        this_feat = df[df.rounded_mass==t].intensity_minsub.values
        # print(t, this_feat.shape)
        # features[t] = this_feat**(1/3)
        features[t] = this_feat

    return features.T

def get_features(row):
    features_path = row['features_path']
    feat_df = pd.read_csv(f'../data/raw/{features_path}')
    derivatized = row['derivatized']

    train_sample_pp = preprocess_sample(feat_df)

    features = int_per_timebin(train_sample_pp)

    return features


if __name__ == '__main__':

    os.makedirs('cache/', exist_ok=True)

    df = pd.read_csv('../data/interim/train_folds.csv')
    df = df.fillna(0)
    # df = df.head(10)

    cache_metadata = {}
    for i, row in tqdm(df.iterrows()):
        path = row['features_path']
        feat = get_features(row)

        cache_metadata[path] = feat

        gc.collect()
        # break
        
    np.save('cache/train_bin005.npy', cache_metadata)

    df = pd.read_csv('../data/raw/metadata.csv')
    df = df.fillna(0)

    val_df = df[df.split == 'val'].reset_index(drop=True)
    print(val_df.shape)
    features_list = Parallel(n_jobs=16, backend="threading")(delayed(get_features)(row) for i, row in tqdm(val_df.iterrows()))
    cache_metadata = {path: feat for path, feat in zip(val_df.features_path.values, features_list)}
    np.save('cache/val_bin005.npy', cache_metadata)

    test_df = df[df.split == 'test'].reset_index(drop=True)
    print(test_df.shape)
    features_list = Parallel(n_jobs=16, backend="threading")(delayed(get_features)(row) for i, row in tqdm(test_df.iterrows()))
    cache_metadata = {path: feat for path, feat in zip(test_df.features_path.values, features_list)}
    np.save('cache/test_bin005.npy', cache_metadata)


