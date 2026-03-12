import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, classification_report, roc_curve
import joblib

from ms_pred import common


# ----------------------------
# YAML -> DataFrame
# ----------------------------
def load_yaml_to_df(yaml_path: str) -> pd.DataFrame:
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    individuals = data.get("individuals", [])
    rows = []

    for ind in individuals:
        row = {
            "ind_recovered": ind.get("ind_recovered", np.nan),
            "mass": ind.get("mass", np.nan),
            "num_peaks_avg": ind.get("num_peaks_avg", np.nan),
            "num_collision_engs": ind.get("num_collision_engs", np.nan),
            "ion": ind.get("ion", None),
        }
        for i in range(1, 51):
            row[f"top_{i}_dist"] = ind.get(f"top_{i}_dist", np.nan)
        if row["ion"] == '[M+H]+' or row["ion"] == '[M-H]-':
            rows.append(row)

    df = pd.DataFrame(rows)

    # Enforce numeric where expected
    for c in ["ind_recovered", "mass", "num_peaks_avg", "num_collision_engs"] + [f"top_{i}_dist" for i in range(1, 51)]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# ----------------------------
# Feature engineering
# ----------------------------
def ion_onehot(ion: str) -> np.ndarray:
    """One-hot with all-zeros fallback for unknown/None ions."""
    n_ions = max(common.ion2onehot_pos.values()) + 1
    vec = np.zeros(n_ions, dtype=np.float32)
    if ion is None:
        return vec
    idx = common.ion2onehot_pos.get(ion)
    if idx is None:
        return vec
    vec[idx] = 1.0
    return vec


def build_feature_matrix(df: pd.DataFrame, topk_features) -> np.ndarray:
    dist_cols = [f"top_{i}_dist" for i in range(1, topk_features + 1)]
    num_cols = dist_cols + ["mass", "num_peaks_avg", "num_collision_engs"]

    X_num = df[num_cols].to_numpy(dtype=np.float32)
    X_ion = np.stack([ion_onehot(x) for x in df["ion"].tolist()], axis=0).astype(np.float32)

    return np.concatenate([X_num, X_ion], axis=1)


# ----------------------------
# Labels: ind_recovered <= k
# ----------------------------
def build_label(df: pd.DataFrame, k: int) -> np.ndarray:
    r = df["ind_recovered"].to_numpy(dtype=np.float32)
    # label 1 if recovered within top-k
    return (r <= float(k)).astype(np.int32)


# ----------------------------
# Training + testing (separate YAMLs)
# ----------------------------
def train_models_and_test(
    train_yaml: str,
    test_yaml: str,
    ks=(1, 3, 5, 10),
    topk_features=20,
):
    train_df = load_yaml_to_df(train_yaml)
    test_df = load_yaml_to_df(test_yaml)

    # Basic row filtering: require core numeric fields present
    required = ["ind_recovered", "mass", "num_peaks_avg", "num_collision_engs"]
    train_df = train_df.dropna(subset=required).copy()
    test_df = test_df.dropna(subset=required).copy()

    models = {}
    reports = {}
    roc_payloads = {}

    for k in ks:
        X_train = build_feature_matrix(train_df, topk_features)
        X_test = build_feature_matrix(test_df, topk_features)

        y_train = build_label(train_df, k=k)
        y_test = build_label(test_df, k=k)

        if len(np.unique(y_train)) < 2:
            reports[k] = {
                "warning": f"Training labels for k={k} have only one class (all {int(y_train[0])}). "
                           f"Model training is not meaningful for this k on the provided train YAML."
            }
            continue
        if len(np.unique(y_test)) < 2:
            # You can still evaluate accuracy, but ROC/PR AUC are undefined for single-class test
            test_single_class = True
        else:
            test_single_class = False

        pipe = Pipeline(
            steps=[
                # Fit imputer/scaler on TRAIN ONLY to avoid leakage
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("clf", LogisticRegression(
                    solver="lbfgs",
                    max_iter=5000,
                    class_weight="balanced",
                )),
            ]
        )

        pipe.fit(X_train, y_train)

        p_test = pipe.predict_proba(X_test)[:, 1]
        yhat_test = (p_test >= 0.5).astype(int)

        acc = accuracy_score(y_test, yhat_test)

        rep = {
            "accuracy@0.5": float(acc),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "positives_train": int(y_train.sum()),
            "positives_test": int(y_test.sum()),
            "classification_report": classification_report(y_test, yhat_test, digits=3),
        }

        if not test_single_class:
            rep["roc_auc"] = float(roc_auc_score(y_test, p_test))
            rep["pr_auc"] = float(average_precision_score(y_test, p_test))
            roc_payloads[k] = {"y_test": y_test, "p_test": p_test}
        else:
            rep["roc_auc"] = None
            rep["pr_auc"] = None
            rep["note"] = "Test labels have a single class; ROC-AUC/PR-AUC are undefined."

        models[k] = pipe
        reports[k] = rep

    return models, reports, roc_payloads


def plot_roc_curves(roc_payload: dict, title: str = "ROC (test set)", topk=10):
    plt.figure()

    d = roc_payload[topk]
    y_test = d["y_test"]
    p_test = d["p_test"]

    fpr, tpr, _ = roc_curve(y_test, p_test)
    auc = roc_auc_score(y_test, p_test)
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")

    # chance line
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, linewidth=0.5)
    plt.tight_layout()
    plt.savefig('iceberg_roc.pdf')


# ----------------------------
# Example CLI-style usage
# ----------------------------
if __name__ == "__main__":
    from pathlib import Path
    train_yaml = "results/dag_inten_nist20/split_1_rnd1/retrieval_nist20_split_1_val_50/rerank_eval_entropy.yaml"
    test_yaml = "results/dag_inten_nist20/split_1_rnd1/retrieval_nist20_split_1_50/rerank_eval_entropy.yaml"
    out_dir = "results/dag_inten_nist20/split_1_rnd1/retrieval_confidence"
    topk_features = 15

    models, reports, roc_payload = train_models_and_test(
        train_yaml,
        test_yaml,
        ks=(1, 3, 5, 10),
        topk_features=topk_features,
    )

    for k in (1, 3, 5, 10):
        print(f"\n===== k={k} (label: ind_recovered <= {k}) =====")
        rep = reports.get(k, {})
        if "warning" in rep:
            print(rep["warning"])
            continue

        roc = rep.get("roc_auc", None)
        pr = rep.get("pr_auc", None)
        if roc is not None and pr is not None:
            print(f"ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f} | Acc@0.5: {rep['accuracy@0.5']:.4f}")
        else:
            print(f"Acc@0.5: {rep['accuracy@0.5']:.4f} (AUCs unavailable)")

        print(f"Train n={rep['n_train']} (pos={rep['positives_train']}), Test n={rep['n_test']} (pos={rep['positives_test']})")
        print(rep["classification_report"])

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    # save one file per k
    for k, pipe in models.items():
        joblib.dump(
            {
                "k": k,
                "topk_features": topk_features,  # must match training
                "pipeline": pipe,  # imputer+scaler+clf
            },
            out_dir / f"lr_conf_k{k}.joblib",
            compress=3,
        )

    if 10 in roc_payload:
        plot_roc_curves(
            roc_payload,
            title="ROC curve for top-10 structural elucidation (NIST'20)",
            topk=10,
        )
