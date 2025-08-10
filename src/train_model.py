import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def main():
    # Load dataset
    df = pd.read_csv('data/iris_with_location.csv')

    # Features and target
    X = df.drop(columns=['species'])
    y = df['species']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Model
    clf = RandomForestClassifier(random_state=42)

    # MLflow tracking
    mlflow.set_experiment("iris_rf_experiment")
    with mlflow.start_run():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Log params, metrics, and model
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("random_state", 42)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(clf, "model")

        print("Accuracy:", acc)
        print("Classification Report:\n", classification_report(y_test, y_pred))

        # Save model with joblib
        joblib.dump(clf, "models/rf_model.joblib")
        print("Saved model to models/rf_model.joblib")

if __name__ == "__main__":
    main()
