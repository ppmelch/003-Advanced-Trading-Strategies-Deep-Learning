from libraries import *

model_name = "PRUEBA1"
model_version = "1"

model = mlflow.tensorflow.load_model(
    model_uri= f'models:/{model_name}/{model_version}'
)

print(model.summary())

