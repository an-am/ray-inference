from typing import Dict

import numpy as np
from ray import serve
from starlette.requests import Request

@serve.deployment
class DeepModel:
    def __init__(self, model_path: str):
        import tensorflow as tf

        self.model_path = model_path
        self.model = tf.keras.models.load_model(model_path)

    async def __call__(self, starlette_request: Request) -> Dict:
        # Step 1: transform HTTP request -> tensorflow input
        # Here we define the request schema to be a json array.
        data = np.array((await starlette_request.json())["array"])
        #data = data.reshape((None,9))

        # Step 2: tensorflow input -> tensorflow output
        prediction = self.model(data)

        # Step 3: tensorflow output -> web output
        return {"prediction": prediction.numpy().tolist(), "file": self.model_path}

DL_model = DeepModel.bind('deep_model.keras')
serve.run(DL_model, route_prefix="/")
