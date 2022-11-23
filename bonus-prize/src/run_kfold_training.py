from loguru import logger
import pandas as pd
from pathlib import Path
import typer
from sklearn.model_selection import StratifiedKFold

from src.make_dataset import make_dataset
from src.training_utils import train_fixed_soup


def main(
        model_dir: Path = typer.Option(
            "./data/processed/", help="Directory to save the output model weights in npy format"
        ),
        features_dir: Path = typer.Option(
            "./data/raw/", help="Path to the raw features"
        ),
        labels_dir: Path = typer.Option(
            "./data/raw/", help="Path to the train_labels csv and val_labels csv"
        ),
        metadata_path: Path = typer.Option(
            "./data/raw/metadata.csv", help="Path to the metadata csv"
        ),
        n_folds: int = typer.Option(
            10, help="Number of folds, Must be at least 2"
        ),
        random_state: int = typer.Option(
            758625225, help="Controls the randomness of each fold"
        ),
        debug: bool = typer.Option(
            False, help="Run on a small subset of the data and a two folds for debugging"
        )
):
    nrows = None
    n_soup_iter = 40
    n_soup = 5
    if debug:
        logger.info("Running in debug mode")
        nrows = 20
        n_folds = 2
        n_soup_iter = 1
        n_soup = 1
        n_folds = 2

    logger.info(f"Loading metadata from {metadata_path}")
    df_meta = pd.read_csv(metadata_path, index_col="sample_id")

    df_meta_full = df_meta[df_meta.split != 'test']
    paths_df = str(features_dir) + "/" + df_meta_full.features_path.astype('str')
    paths_df = paths_df.iloc[:nrows]

    logger.info(f"Loading labels from {labels_dir}")
    df_labels_train = pd.read_csv(labels_dir / 'train_labels.csv',
                                  index_col='sample_id')
    df_labels_val = pd.read_csv(labels_dir / 'val_labels.csv',
                                index_col='sample_id')
    df_labels_full = pd.concat([df_labels_train, df_labels_val])

    logger.info(f"Processing data. Data size: {paths_df.shape[0]}")
    X, Y = make_dataset(paths_df, df_labels_full)

    logger.info("Training models")
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for i, (train_index, test_index) in enumerate(kf.split(X, Y[:, 4])):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        f = model_dir / f'model-soup-{i}'
        logger.info(f"Training model {i}")
        train_fixed_soup(x_train, x_test, y_train, y_test, f,
                         n_iter=n_soup_iter, n_soup=n_soup)

    logger.info(f"Completed Training models to {model_dir}")


if __name__ == "__main__":
    typer.run(main)

