from loguru import logger
import pandas as pd
from pathlib import Path
import typer

from src.make_dataset import make_dataset
from src.training_utils import train_fixed_soup


def main(
        model_file: Path = typer.Option(
            "./data/processed/single-model", help="File to the output model weights in npy format"
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
        debug: bool = typer.Option(
            False, help="Run on a small subset of the data and a two folds for debugging"
        )
):
    nrows = None
    n_soup_iter = 40
    n_soup = 5
    if debug:
        logger.info("Running in debug mode")
        nrows = 10
        n_soup_iter = 1
        n_soup = 1

    logger.info(f"Loading metadata from {metadata_path}")
    df_meta = pd.read_csv(metadata_path, index_col="sample_id")

    # train split
    df_meta_train = df_meta[df_meta.split == 'train']
    paths_df = str(features_dir) + "/" + df_meta_train.features_path.astype('str')
    df_paths_train = paths_df.iloc[:nrows]

    # val split
    df_meta_val = df_meta[df_meta.split == 'val']
    paths_df = str(features_dir) + "/" + df_meta_val.features_path.astype('str')
    df_paths_val = paths_df.iloc[:nrows]

    logger.info(f"Loading labels from {labels_dir}")
    df_labels_train = pd.read_csv(labels_dir / 'train_labels.csv',
                                  index_col='sample_id')
    df_labels_val = pd.read_csv(labels_dir / 'val_labels.csv',
                                index_col='sample_id')

    # create the datasets
    logger.info(f"Processing training data. Data size: {df_paths_train.shape[0]}")
    x_train, y_train = make_dataset(df_paths_train, df_labels_train)
    logger.info(f"Processing val data. Data size: {df_paths_val.shape[0]}")
    x_val, y_val = make_dataset(df_paths_val, df_labels_val)

    # train
    logger.info("Training model")
    train_fixed_soup(x_train, x_val, y_train, y_val, model_file,
                     n_iter=n_soup_iter, n_soup=n_soup)

    logger.info(f"Completed Training models to {model_file}")


if __name__ == "__main__":
    typer.run(main)

