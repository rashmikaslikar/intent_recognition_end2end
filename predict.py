import json
from pathlib import Path
from typing import Any, Dict, Iterable, List
from urllib.parse import urlparse
import pandas as pd

import numpy as np
import ray
import typer
from numpyencoder import NumpyEncoder
from ray.air import Result
from ray.train.torch.torch_checkpoint import TorchCheckpoint
from typing_extensions import Annotated

from config import logger, mlflow
from data import CustomPreprocessor
from models import BERT_Arch 
from utils import collate_fn

# Initialize Typer CLI app
app = typer.Typer()

sorted_runs = mlflow.search_runs(
        experiment_names=['bert'],
        order_by=[f"metrics.{'val_loss'} {'ASC'}"],
    )
#print(sorted_runs)
run_id = sorted_runs.iloc[0].run_id
#print(run_id)




def decode(indices: Iterable[Any], index_to_class: Dict) -> List:
    """Decode indices to labels.

    Args:
        indices (Iterable[Any]): Iterable (list, array, etc.) with indices.
        index_to_class (Dict): mapping between indices and labels.

    Returns:
        List: list of labels.
    """
    return [index_to_class[index] for index in indices]

def format_prob(prob: Iterable, index_to_class: Dict) -> Dict:
    """Format probabilities to a dictionary mapping class label to probability.

    Args:
        prob (Iterable): probabilities.
        index_to_class (Dict): mapping between indices and labels.

    Returns:
        Dict: Dictionary mapping class label to probability.
    """
    d = {}
    for i, item in enumerate(prob):
        d[index_to_class[i]] = item
    return d

class TorchPredictor:
    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model
        self.model.eval()
        
    def __call__(self, batch):
        results = self.model.predict(collate_fn(batch))
        return {"output": results}

    def predict_proba(self, batch):
        results = self.model.predict_proba(collate_fn(batch))
        return {"output": results}
        
    def get_preprocessor(self):
        return self.preprocessor
        
    @classmethod
    def from_checkpoint(cls, checkpoint):
        metadata = checkpoint.get_metadata()
        preprocessor = CustomPreprocessor(intent_to_index=metadata["intent_to_index"])
        model = BERT_Arch.load(Path(checkpoint.path, "args.json"), Path(checkpoint.path, "model.pt"))
        return cls(preprocessor=preprocessor, model=model)
    
def predict_proba(
    ds: ray.data.dataset.Dataset,
    predictor: TorchPredictor,
) -> List:  # pragma: no cover, tested with inference workload
    """Predict tags (with probabilities) for input data from a dataframe.

    Args:
        df (pd.DataFrame): dataframe with input features.
        predictor (TorchPredictor): loaded predictor from a checkpoint.

    Returns:
        List: list of predicted labels.
    """
    preprocessor = predictor.get_preprocessor()
    preprocessed_ds = preprocessor.transform(ds,'val')
    outputs = preprocessed_ds.map_batches(predictor.predict_proba)
    print(outputs)
    
    y_prob = np.array([d["output"] for d in outputs.take_all()])
    results = []
    for i, prob in enumerate(y_prob):
        tag = preprocessor.index_to_intent[prob.argmax()]
        results.append({"prediction": tag, "probabilities": format_prob(prob, preprocessor.index_to_intent)})
    return results

#@app.command()
def get_best_run_id(experiment_name: str = "", metric: str = "", mode: str = "") -> str:  # pragma: no cover, mlflow logic
    """Get the best run_id from an MLflow experiment.

    Args:
        experiment_name (str): name of the experiment.
        metric (str): metric to filter by.
        mode (str): direction of metric (ASC/DESC).

    Returns:
        str: best run id from experiment.
    """
    sorted_runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=[f"metrics.{metric} {mode}"],
    )
    run_id = sorted_runs.iloc[0].run_id
    print(run_id)
    return run_id


def get_best_checkpoint(run_id: str) -> TorchCheckpoint:  # pragma: no cover, mlflow logic
    """Get the best checkpoint from a specific run.

    Args:
        run_id (str): ID of the run to get the best checkpoint from.

    Returns:
        TorchCheckpoint: Best checkpoint from the run.
    """
    #artifact_dir = urlparse(mlflow.get_run(run_id).info.artifact_uri).path  # get path from mlflow
    artifact_dir = mlflow.get_run(run_id).info.artifact_uri
    artifact_dir = artifact_dir.replace('file://', '')
    artifact_dir = artifact_dir.replace('/', '\\')
    results = Result.from_path(artifact_dir)
    return results.best_checkpoints[0][0]

@app.command()
def predict(
    run_id: Annotated[str, typer.Option(help="id of the specific run to load from")] = run_id,
    date: Annotated[str, typer.Option(help="project title")] = '2024-06-20',
    search_query: Annotated[str, typer.Option(help="project description")] = 'übersetzer',
    market: Annotated[str, typer.Option(help="project description")] = 'de-de',
    geo_country: Annotated[str, typer.Option(help="project description")] = 'DE',
    device_type: Annotated[str, typer.Option(help="project description")] = 'Mobile',
    browser_name: Annotated[str, typer.Option(help="project description")] = 'Safari',
    intent: Annotated[str, typer.Option(help="project description")] = 'OTHER',
    daily_query_count: Annotated[str, typer.Option(help="project description")] = 2,
) -> Dict:  # pragma: no cover, tested with inference workload
    """Predict the tag for a project given it's title and description.

    Args:
        run_id (str): id of the specific run to load from. Defaults to None.
        title (str, optional): project title. Defaults to "".
        description (str, optional): project description. Defaults to "".

    Returns:
        Dict: prediction results for the input data.
    """
    # Load components
    best_checkpoint = get_best_checkpoint(run_id=run_id)
    predictor = TorchPredictor.from_checkpoint(best_checkpoint)
    df =pd.DataFrame({"date": [date],
            "search_query": [search_query],
            "market": [market],
            "geo_country": [geo_country],
            "device_type": [device_type],
            "browser_name": [browser_name],
            "intent": [intent],
            "daily_query_count": [daily_query_count],
            })

    # Predict
    sample_ds = ray.data.from_pandas(df)
    results = predict_proba(ds=sample_ds, predictor=predictor)
    logger.info(json.dumps(results, cls=NumpyEncoder, indent=2))
    return results


if __name__ == "__main__":  # pragma: no cover, application
    app()
