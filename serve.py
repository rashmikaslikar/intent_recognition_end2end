import argparse
import os
from http import HTTPStatus
from typing import Dict
import time

import ray
from fastapi import FastAPI
from ray import serve
from starlette.requests import Request

import evaluate, predict
from config import MLFLOW_TRACKING_URI, mlflow

# Define application
app = FastAPI(
    title="Intent Recognition",
    description="Classify query intents",
    version="0.1",
)

sorted_runs = mlflow.search_runs(
        experiment_names=['bert'],
        order_by=[f"metrics.{'val_loss'} {'ASC'}"],
    )
#print(sorted_runs)
run_id = sorted_runs.iloc[0].run_id
#print(run_id)

@serve.deployment(num_replicas="1", ray_actor_options={"num_cpus": 1, "num_gpus": 0})
@serve.ingress(app)
class ModelDeployment:
    def __init__(self, run_id: str, threshold: int = 0.9):
        """Initialize the model."""
        self.run_id = run_id
        self.threshold = threshold
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)  # so workers have access to model registry
        best_checkpoint = predict.get_best_checkpoint(run_id=run_id)
        self.predictor = predict.TorchPredictor.from_checkpoint(best_checkpoint)

    @app.post("/predict/")
    async def _predict(self, request: Request):
        data = await request.json()
        sample_ds = ray.data.from_items([{"date": data.get("date", ""),
                                          "search_query": data.get("search_query", ""),
                                          "market": data.get("market", ""),
                                          "geo_country": data.get("geo_country", ""),
                                          "device_type": data.get("device_type", ""),
                                          "browser_name": data.get("browser_name", ""),
                                          "daily_query_count": data.get("daily_query_count", ""),
                                          }])
        print(sample_ds.count())
        results = predict.predict_proba(ds=sample_ds, predictor=self.predictor)

        # Apply custom logic
        for i, result in enumerate(results):
            pred = result["prediction"]
            prob = result["probabilities"]
            if prob[pred] < self.threshold:
                results[i]["prediction"] = "other"

        return {"results": results}
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", help="run ID to use for serving.")
    parser.add_argument("--threshold", type=float, default=0.9, help="threshold for `other` class.")
    args = parser.parse_args()
    # Run service
    ray.init()
    serve.run(ModelDeployment.bind(run_id=run_id, threshold=args.threshold))
    # Keep the server running
    try:
        while True:
            time.sleep(3600)  # Sleep for 1 hour
    except KeyboardInterrupt:
        print("Shutting down...")