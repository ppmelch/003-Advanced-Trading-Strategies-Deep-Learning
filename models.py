from libraries import *



# Model MLP

def model_MLP(X_train_scaled, X_test_scaled, X_validation_scaled, y_train, y_test, y_validation):

    mlp = MLPClassifier(hidden_layer_sizes=(64, 32),
                        activation='relu',
                        solver='adam',
                        max_iter=10000)
    mlp.fit(X_train_scaled, y_train)
    y_pred = mlp.predict(X_test_scaled)
    y_val_pred = mlp.predict(X_validation_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    val_accuracy = accuracy_score(y_validation, y_val_pred)
    class_report = classification_report(y_test, y_pred)

    return accuracy, val_accuracy, class_report


# Model CNN
'''
def model_CNN(X_train, X_test, X_validation, y_train, y_test, y_validation):
    mlflow.tensorflow.autolog()
    mlflow.set_experiment("CNN MLP Model v1")

    def model(params):
        model = tf.keras.models.Sequential()

    pass

    '''



def model_CNN(X_scaled, y, lookback=10, params=None, name="CNN Trading"):
    """
    Función modular para entrenar un modelo CNN 1D para predicción de subida/bajada de precio.
    """
    
    if params is None:
        params = {
            "conv_layers": 2,
            "filters": 32,
            "kernel_size": 3,
            "dense_units": 64,
            "activation": "relu",
            "optimizer": "adam",
            "epochs": 10,
            "batch_size": 32
        }
    
    # =========================
    # Preparar datos en ventanas
    # =========================
    X_cnn = np.array([X_scaled[i-lookback:i] for i in range(lookback, len(X_scaled))])
    y_cnn = np.array(y[lookback:])
    
    # =========================
    # Definir modelo CNN 1D
    # =========================
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(lookback, X_scaled.shape[1])))
    
    filters = params["filters"]
    for _ in range(params["conv_layers"]):
        model.add(tf.keras.layers.Conv1D(filters=filters,
                                         kernel_size=params["kernel_size"],
                                         activation=params["activation"]))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
        filters *= 2
    
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(params["dense_units"], activation=params["activation"]))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    
    model.compile(optimizer=params["optimizer"], loss="binary_crossentropy", metrics=["accuracy"])
    
    # =========================
    # MLflow setup
    # =========================
    mlflow.tensorflow.autolog()
    mlflow.set_experiment(name)
    
    input_example = np.array([X_scaled[:lookback]])  # shape (1, lookback, features)
    
    with mlflow.start_run() as run:
        history = model.fit(X_cnn, y_cnn, 
                            epochs=params["epochs"], 
                            batch_size=params["batch_size"], 
                            validation_split=0.2,
                            verbose=2)
        
        final_metrics = {
            "val_loss": history.history["val_loss"][-1],
            "val_accuracy": history.history["val_accuracy"][-1]
        }
        
        mlflow.log_metrics(final_metrics)
        mlflow.tensorflow.log_model(model, name="model", input_example=input_example)
    
    return model, final_metrics
