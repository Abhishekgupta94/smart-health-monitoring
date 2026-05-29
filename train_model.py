import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "heart_model.pkl"
SCALER_PATH = PROJECT_DIR / "scaler.pkl"
DATASET_CANDIDATES = [
    PROJECT_DIR / "cleaned_merged_heart_dataset.csv",
    PROJECT_DIR.parent / "cleaned_merged_heart_dataset.csv",
]

FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalachh",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
TARGET_COLUMN = "target"


def resolve_dataset_path():
    for candidate in DATASET_CANDIDATES:
        if candidate.exists():
            return candidate

    checked_paths = "\n".join(str(path) for path in DATASET_CANDIDATES)
    raise FileNotFoundError(
        "Dataset not found. Checked these locations:\n"
        f"{checked_paths}\n\n"
        "Move `cleaned_merged_heart_dataset.csv` into the project folder "
        "or the parent Project folder and run again."
    )


def main():
    dataset_path = resolve_dataset_path()

    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)

    missing_columns = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    df = df[FEATURE_COLUMNS + [TARGET_COLUMN]].dropna()

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    print("\nTraining complete.")
    print(f"Rows used: {len(df)}")
    print(f"Train rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(matrix)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(model, model_file)

    with open(SCALER_PATH, "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)

    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved scaler to: {SCALER_PATH}")
    print("\nNext step: run `python app.py` to use the trained model.")


if __name__ == "__main__":
    main()
