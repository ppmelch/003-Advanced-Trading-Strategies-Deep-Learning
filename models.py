from libraries import *



# Model MLP

def model_MLP(X_train_scaled, X_test_scaled, X_validation_scaled, y_train, y_test, y_validation):

    mlp = MLPClassifier(hidden_layer_sizes=(64, 32),
                        activation='relu',
                        solver='adam',
                        max_iter=200)
    mlp.fit(X_train_scaled, y_train)
    y_pred = mlp.predict(X_test_scaled)
    y_val_pred = mlp.predict(X_validation_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    val_accuracy = accuracy_score(y_validation, y_val_pred)
    class_report = classification_report(y_test, y_pred)

    return accuracy, val_accuracy, class_report



