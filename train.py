import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


DATA_PATH = "data/SpotifyAudioFeaturesApril2019.csv"
OUTPUT_MODEL_PATH = "model.bin"

HIT_THRESHOLD = 70
RANDOM_STATE = 1

COLS_TO_DROP = [
    "track_id",
    "track_name",
    "artist_name",
    "album_name",
    "popularity"
]

def metrics_dict(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.lower()

    # Label: hit if popularity >= 70
    df["hit"] = (df["popularity"] >= HIT_THRESHOLD).astype(int)
    # Drop columns
    df = df.drop(columns=COLS_TO_DROP, errors="ignore")
    return df


def split_data(df):
    df_full_train, df_test = train_test_split(
        df, test_size=0.2, stratify=df["hit"], random_state=RANDOM_STATE
    )
    df_train, df_val = train_test_split(
        df_full_train, test_size=0.25, stratify=df_full_train["hit"], random_state=RANDOM_STATE
    )
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    return df_train, df_val, df_test


def make_xy(df_train, df_val, df_test):
    y_train = df_train.hit.values
    y_val = df_val.hit.values
    y_test = df_test.hit.values

    features = [c for c in df_train.columns if c != "hit"]

    X_train = df_train[features]
    X_val = df_val[features]
    X_test = df_test[features]

    return X_train, X_val, X_test, y_train, y_val, y_test, features


def scale(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    return scaler, X_train_s, X_val_s, X_test_s


def tune_logistic_regression(X_train, y_train, X_val, y_val):
    C_list = [0.01, 0.1, 1, 10, 100]
    best_model = None
    best_params = None
    best_metrics = None

    for C in C_list:
        model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, C=C)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        m = metrics_dict(y_val, pred)

        if best_metrics is None or m["f1"] > best_metrics["f1"]:
            best_model = model
            best_params = {"C": C, "max_iter": 2000, "random_state": RANDOM_STATE}
            best_metrics = m

    return best_model, best_params, best_metrics


def tune_random_forest(X_train, y_train, X_val, y_val):
    n_estimators_list = [200, 500]
    max_depth_list = [None, 10, 20]
    min_samples_leaf_list = [1, 3]
    class_weight_list = [None, "balanced"]

    best_model = None
    best_params = None
    best_metrics = None

    for n_estimators in n_estimators_list:
        for max_depth in max_depth_list:
            for min_leaf in min_samples_leaf_list:
                for cw in class_weight_list:
                    model = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_leaf=min_leaf,
                        class_weight=cw,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    )
                    model.fit(X_train, y_train)
                    pred = model.predict(X_val)
                    m = metrics_dict(y_val, pred)

                    if best_metrics is None or m["f1"] > best_metrics["f1"]:
                        best_model = model
                        best_params = {
                            "n_estimators": n_estimators,
                            "max_depth": max_depth,
                            "min_samples_leaf": min_leaf,
                            "class_weight": cw,
                            "random_state": RANDOM_STATE,
                            "n_jobs": -1,
                        }
                        best_metrics = m

    return best_model, best_params, best_metrics


def tune_gradient_boosting(X_train, y_train, X_val, y_val):
    n_estimators_list = [100, 200, 300]
    learning_rate_list = [0.05, 0.1, 0.2]
    subsample_list = [1.0, 0.8]

    best_model = None
    best_params = None
    best_metrics = None

    for n_estimators in n_estimators_list:
        for lr in learning_rate_list:
            for subsample in subsample_list:
                model = GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    learning_rate=lr,
                    subsample=subsample,
                    random_state=RANDOM_STATE,
                )
                model.fit(X_train, y_train)
                pred = model.predict(X_val)
                m = metrics_dict(y_val, pred)

                if best_metrics is None or m["f1"] > best_metrics["f1"]:
                    best_model = model
                    best_params = {
                        "n_estimators": n_estimators,
                        "learning_rate": lr,
                        "subsample": subsample,
                        "random_state": RANDOM_STATE,
                    }
                    best_metrics = m

    return best_model, best_params, best_metrics


def main():
    df = load_data()
    df_train, df_val, df_test = split_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test, features = make_xy(df_train, df_val, df_test)

    scaler, X_train_s, X_val_s, X_test_s = scale(X_train, X_val, X_test)

    lr_model, lr_params, lr_val_metrics = tune_logistic_regression(X_train_s, y_train, X_val_s, y_val)
    rf_model, rf_params, rf_val_metrics = tune_random_forest(X_train_s, y_train, X_val_s, y_val)
    gb_model, gb_params, gb_val_metrics = tune_gradient_boosting(X_train_s, y_train, X_val_s, y_val)

    candidates = [
        ("LogisticRegression", lr_model, lr_params, lr_val_metrics),
        ("RandomForest", rf_model, rf_params, rf_val_metrics),
        ("GradientBoosting", gb_model, gb_params, gb_val_metrics),
    ]
    candidates.sort(key=lambda x: x[3]["f1"], reverse=True)

    best_name, best_model, best_params, best_val_metrics = candidates[0]
    test_pred = best_model.predict(X_test_s)
    test_metrics = metrics_dict(y_test, test_pred)

    print("=== Best Model (by Val F1) ===")
    print("Model:", best_name)
    print("Best params:", best_params)
    print("Val metrics:", best_val_metrics)
    print("\n=== Test ===")
    print("Test metrics:", test_metrics)
    print("\nTest report:\n", classification_report(y_test, test_pred, digits=4))

    # Save model.bin
    model_data = {
        "model": best_model,
        "scaler": scaler,
        "metrics": test_metrics,
        "feature_names": features,
        "config": {
            "hit_threshold": HIT_THRESHOLD,
            "random_state": RANDOM_STATE,
            "data_path": DATA_PATH,
            "best_model": best_name,
            "best_params": best_params,
        }
    }

    with open(OUTPUT_MODEL_PATH, "wb") as f:
        pickle.dump(model_data, f)
    print(f"\nSaved: {OUTPUT_MODEL_PATH}")


if __name__ == "__main__":
    main()
