from loguru import logger
import pandas as pd
from pathlib import Path
import typer
from src.model import get_model
from src.make_dataset import make_dataset


def main(
    model_dir: Path = typer.Option(
        "./models/", help="Directory of the saved model weights"
    ),
    features_path: Path = typer.Option(
        "./data/raw/", help="Path to the raw features"
    ),
    submission_save_path: Path = typer.Option(
        "./data/processed/submission.csv", help="Path to save the generated submission"
    ),
    submission_format_path: Path = typer.Option(
        "./data/raw/submission_format.csv", help="Path to save the submission format csv"
    ),
    metadata_path: Path = typer.Option(
        "./data/raw/metadata.csv", help="Path to the metadata csv"
    ),
    debug: bool = typer.Option(
        False, help="Run on a small subset of the data and a single model for debugging"
    )
):
    n_rows = None
    n_models = 10
    if debug:
        logger.info("Running in debug mode")
        n_rows = 10
        n_models = 1

    logger.info(f"Loading feature data from {features_path}")
    df_meta = pd.read_csv(metadata_path, index_col="sample_id")

    df_meta_test = df_meta[df_meta.split != 'train']
    paths_df = str(features_path) + "/" + df_meta_test.features_path.astype('str')
    paths_df = paths_df.iloc[:n_rows]

    logger.info(f"Processing feature data. size: {paths_df.shape[0]}")
    X = make_dataset(paths_df)

    logger.info("Creating model")
    model = get_model()

    logger.info("Predicting labels")
    probas = 0
    for i in range(n_models):
        h5_path = model_dir / f"s{i}.h5"
        model.load_weights(h5_path)
        logger.info(f"Loading trained model weights from {h5_path}")

        probas += model.predict(X)
    probas /= n_models

    # generate submission
    my_submission = pd.read_csv(submission_format_path)
    my_submission = my_submission.iloc[:n_rows]
    my_submission["sample_id"] = paths_df.index.values

    for i, c in enumerate(my_submission.columns[1:]):
        my_submission[c] = probas[:, i]

    my_submission.to_csv(submission_save_path, index=False)
    logger.success(f"Submission saved to {submission_save_path}")


if __name__ == "__main__":
    typer.run(main)

