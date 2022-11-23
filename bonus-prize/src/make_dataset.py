import pandas as pd
import numpy as np
from typing import Optional


MZ_SUB = [6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
          20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
          33, 34, 35, 36, 37, 39, 40, 41, 42, 43, 44, 45, 46,
          47, 48, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
          61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73,
          74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86,
          87, 88, 89, 90, 91, 92, 94, 98, 100, 103, 104, 105, 106,
          107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
          120, 121, 122, 123, 125, 126, 128, 129, 130, 131, 132, 133, 134,
          135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 147, 148,
          149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161,
          162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174,
          175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187,
          188, 189, 190, 191, 192, 193, 194, 196, 197, 198, 199, 200, 201,
          202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214,
          215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227,
          228, 229, 230, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241,
          242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254,
          255, 256, 257, 258, 259, 261, 262, 263, 264, 265, 266, 267, 268,
          269, 270, 271, 272, 273, 274, 275, 276, 278, 279, 280, 281, 282,
          283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295,
          296, 297, 298, 299, 300, 301, 302, 303, 304, 306, 307, 308, 310,
          311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323,
          324, 325, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337,
          338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349]


def create_peak_detection_feature(df, mz_subset):
    m = df.mass.values
    mass_cycles = np.digitize(list(range(len(df))),
                              np.where(m[1:] - m[:-1] < 0)[0])
    max_mass_times = []
    max_mass_value = []

    for m in mz_subset:

        x = df[m == df.mass]
        indexes = x.index.values
        t, v = -1, -1
        if x.shape[0] > 0:
            intensity = x.intensity.values
            idx = indexes[np.argmax(intensity)]
            t = mass_cycles[idx] / mass_cycles.max()
            v = np.log1p(intensity.max()) / 10
        max_mass_times.append(t)
        max_mass_value.append(v)

    return np.array(max_mass_times + max_mass_value, 'float32')


def add_diff_features(df):
    intensity = df.intensity.values

    dx = intensity[1:] - intensity[:-1]
    df['dx'] = [0] + list(dx / 1000)

    dx = intensity[2:] - intensity[:-2]
    df['dxx'] = [0, 0] + list(dx / 1000)

    dx = intensity[1:] - intensity[:-1]
    dxdx = dx[1:] - dx[:-1]
    df['dxdx'] = [0, 0] + list(dxdx / 10)

    return df


def create_sample(features_df, mz_subset):
    x = features_df

    x.mass = x.mass.transform(round)

    mmc = create_peak_detection_feature(x, mz_subset)

    x = x[x.mass != 4]

    intensity = x.intensity.values
    intensity = intensity - intensity.min()

    x = add_diff_features(x)

    m = x.groupby(
        pd.cut(x["mass"], np.arange(0, 350 + 1, 1))
    ).max()[['intensity']].iloc[mz_subset].fillna(0).values.reshape(-1)
    m /= m.max()

    x = x.groupby(
        pd.cut(x["mass"], np.arange(0, 350 + 1, 1))
    ).mean()[['intensity', 'dx', 'dxx', 'dxdx']].iloc[mz_subset].fillna(0)

    intensity = x.intensity.values
    dx = x.dx.values
    dxx = x.dxx.values
    dxdx = x.dxdx.values

    # trasnform
    intensity = np.log1p(intensity)

    dx = np.sign(dx) * np.log1p(np.abs(dx))
    dxx = np.sign(dxx) * np.log1p(np.abs(dxx))
    dxdx = np.sign(dxdx) * np.log1p(np.abs(dxdx))
    return np.concatenate((intensity, dx, mmc, dxx, m, dxdx))


def make_dataset(paths_df: pd.DataFrame, df_labels: Optional[pd.DataFrame] = None):
    features, labels = [], []

    ids_, paths = paths_df.index, paths_df.values
    for sample_id, sample_path in zip(ids_, paths):
        x = pd.read_csv(sample_path)
        x = create_sample(x, MZ_SUB)
        features.append(x)
        if df_labels is not None:
            y = df_labels.loc[sample_id].values
            labels.append(y)

    features = np.array(features, np.float32)
    if df_labels is not None:
        return features, np.array(labels)
    return features

