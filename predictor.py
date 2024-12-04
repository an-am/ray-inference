from ray import serve
from starlette.requests import Request
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from typing import Dict

@serve.deployment(num_replicas=2, ray_actor_options={'resources': {"predictors_node": 1}})
class Predictor:
    def __init__(self):
        tf.config.experimental.set_visible_devices([], "GPU")
        self.model = load_model('deep_model.h5')

    async def __call__(self, starlette_request: Request) -> Dict:
        input_array = np.array((await starlette_request.json())["array"])
        reshaped_array = input_array.reshape((1, 9))
        prediction = self.model.predict(reshaped_array, verbose=0)
        return {"prediction": prediction[0][0]}

predictor_app = Predictor.bind()