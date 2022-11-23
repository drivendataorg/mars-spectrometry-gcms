import pandas as pd
from sklearn import model_selection
from tqdm.auto import tqdm
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

def create_folds(data, num_splits, seed=42):
    mskf = MultilabelStratifiedKFold(n_splits=num_splits, shuffle=True, random_state=seed)
    labels = ['aromatic', 'hydrocarbon', 'carboxylic_acid',
       'nitrogen_bearing_compound', 'chlorine_bearing_compound',
       'sulfur_bearing_compound', 'alcohol', 'other_oxygen_bearing_compound',
       'mineral']
    data_labels = data[labels].values

    for f, (t_, v_) in enumerate(mskf.split(data, data_labels)):
        data.loc[v_, "fold"] = f

    return data

for seed in [42,2]:
	suf = ''
	if seed == 2:
		suf = '_s2'
	df = pd.read_csv("data/raw/metadata.csv")
	df = df[df.split == 'train']
	print(df.shape)

	label = pd.read_csv('data/raw/train_labels.csv')

	df = df.merge(label, on=['sample_id'])

	print(df.shape)

	# print(df.columns)

	# print(df.head())
	df = create_folds(df, num_splits=5, seed=seed)
	df['fold'] = df['fold'].astype(int)

	df.to_csv(f"data/interim/train_folds{suf}.csv", index=False)
	print("Folds created successfully")


	df = pd.read_csv("data/raw/metadata.csv")
	df = df[df.split == 'val']
	print(df.shape)

	label = pd.read_csv('data/raw/val_labels.csv')

	df = df.merge(label, on=['sample_id'])
	# df['fold'] = -1 
	df = create_folds(df, num_splits=5, seed=seed)
	df['fold'] = df['fold'].astype(int)

	df.to_csv(f"data/interim/val_folds{suf}.csv", index=False)

