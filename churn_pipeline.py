"""Telco customer churn prediction: end-to-end pipeline.

Run with:  python churn_pipeline.py

Steps
  1. Load and clean the data (documented decisions, no silent fixes)
  2. Hold out a stratified test set (touched once, for reporting only)
  3. Compare models with 5-fold CV on the training set
  4. Ablation: do the engineered features actually help?
  5. Tune each model, pick the winner by CV (never by test score)
  6. Pick a decision threshold from out-of-fold training predictions
  7. Final evaluation on the test set + bootstrap confidence intervals
  8. Error analysis by segment, permutation importance, plots

Everything is seeded, so re-running reproduces the numbers in results/.
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter
from scipy.stats import loguniform
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from churn_features import add_features

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

SEED = 42
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Telco-Customer-Churn.csv"
RESULTS = ROOT / "results"
MODELS_DIR = ROOT / "models"

# Chart palette (validated categorical slots 1-2, light surface)
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
BLUE_RAMP = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#3987e5", "#0d366b"])

SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
}


# ----------------------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_PATH)

    # TotalCharges is stored as text; 11 rows are blank. All 11 have tenure == 0,
    # i.e. brand-new customers who have not been billed yet, so 0 is the right fill.
    total = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce")
    blank = total.isna()
    assert (df.loc[blank, "tenure"] == 0).all(), "blank TotalCharges with tenure > 0"
    df["TotalCharges"] = total.fillna(0.0)

    assert not df["customerID"].duplicated().any()
    y = (df.pop("Churn") == "Yes").astype(int)
    X = df.drop(columns="customerID")  # an ID carries no signal and invites leakage
    return X, y


# ----------------------------------------------------------------------------
# 2. Models
# ----------------------------------------------------------------------------
def build_pipeline(model, engineer=True):
    """Raw dataframe in, probability out. Feature engineering lives inside the
    pipeline, so the saved model can score raw customer rows directly."""
    prep = ColumnTransformer(
        [
            ("num", StandardScaler(), make_column_selector(dtype_include="number")),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                make_column_selector(dtype_exclude="number"),
            ),
        ]
    )
    return Pipeline(
        [
            ("features", FunctionTransformer(add_features if engineer else None)),
            ("prep", prep),
            ("model", model),
        ]
    )


def make_models():
    return {
        "Dummy (class prior)": DummyClassifier(strategy="prior"),
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced", n_jobs=1, random_state=SEED
        ),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=SEED),
    }


SEARCH_SPACES = {
    "Logistic Regression": ("grid", {"model__C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]}),
    "Random Forest": (
        "random",
        {
            "model__max_depth": [4, 6, 8, 12, None],
            "model__min_samples_leaf": [1, 3, 5, 10, 20],
            "model__max_features": ["sqrt", 0.3, 0.5],
        },
    ),
    "Hist Gradient Boosting": (
        "random",
        {
            "model__learning_rate": loguniform(0.01, 0.2),
            "model__max_iter": [100, 200, 300, 500],
            "model__max_leaf_nodes": [8, 15, 31],
            "model__min_samples_leaf": [20, 40, 80],
            "model__l2_regularization": loguniform(1e-3, 10),
        },
    ),
}


def cv_row(name, pipe, X, y, cv):
    res = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
    row = {"model": name}
    for k in SCORING:
        row[f"{k}_mean"] = res[f"test_{k}"].mean()
        row[f"{k}_std"] = res[f"test_{k}"].std()
    return row


def tune(name, model, X, y, cv):
    kind, space = SEARCH_SPACES[name]
    pipe = build_pipeline(model, engineer=True)
    if kind == "grid":
        search = GridSearchCV(pipe, space, scoring="average_precision", cv=cv, n_jobs=-1)
    else:
        n_iter = 25 if name == "Hist Gradient Boosting" else 15
        search = RandomizedSearchCV(
            pipe, space, n_iter=n_iter, scoring="average_precision", cv=cv, n_jobs=-1, random_state=SEED
        )
    search.fit(X, y)
    params = {k.replace("model__", ""): (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()}
    return search.best_estimator_, search.best_score_, params


# ----------------------------------------------------------------------------
# 3. Metrics helpers
# ----------------------------------------------------------------------------
def threshold_metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    return {
        "threshold": float(thr),
        "accuracy": float((pred == y).mean()),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }


def best_f1_threshold(y, p):
    grid = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def capture_rate(y, p, frac):
    """Share of all churners found when contacting the top `frac` riskiest customers."""
    k = int(np.ceil(frac * len(y)))
    return float(y[np.argsort(-p)[:k]].sum() / y.sum())


def bootstrap_ci(y, p, n_boot=1000):
    rng = np.random.default_rng(SEED)
    aucs, aps = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        aucs.append(roc_auc_score(y[i], p[i]))
        aps.append(average_precision_score(y[i], p[i]))
    return {
        "roc_auc_ci95": [float(v) for v in np.percentile(aucs, [2.5, 97.5])],
        "pr_auc_ci95": [float(v) for v in np.percentile(aps, [2.5, 97.5])],
    }


def segment_report(df_te, y, p, thr, col):
    rows = []
    pred = (p >= thr).astype(int)
    for level, idx in df_te.groupby(col).indices.items():
        yt, pr = y[idx], pred[idx]
        rows.append(
            {
                "segment_type": col,
                "segment": level,
                "n": len(idx),
                "actual_churn_rate": float(yt.mean()),
                "predicted_churn_rate": float(pr.mean()),
                "precision": float(precision_score(yt, pr, zero_division=0)),
                "recall": float(recall_score(yt, pr, zero_division=0)),
                "roc_auc": float(roc_auc_score(yt, p[idx])) if 0 < yt.sum() < len(yt) else np.nan,
            }
        )
    return rows


# ----------------------------------------------------------------------------
# 4. Plots
# ----------------------------------------------------------------------------
def style_axes(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, length=0, labelsize=10)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def new_fig(*args, **kwargs):
    fig, axes = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes


def plot_eda(X, y, path):
    d = add_features(X).assign(churn=y.values)
    overall = y.mean()
    panels = [
        ("Contract", "Contract type", ["Month-to-month", "One year", "Two year"]),
        ("tenure_bucket", "Tenure (months)", ["0-6", "7-12", "13-24", "25-48", "49+"]),
        ("InternetService", "Internet service", ["Fiber optic", "DSL", "No"]),
    ]
    fig, axes = new_fig(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, (col, title, order) in zip(axes, panels):
        rate = d.groupby(col)["churn"].mean().reindex(order)
        ax.bar(order, rate.values, color=BLUE, width=0.55)
        ax.axhline(overall, color=MUTED, linestyle="--", linewidth=1)
        for i, v in enumerate(rate.values):
            ax.text(i, v + 0.012, f"{v:.0%}", ha="center", color=INK2, fontsize=10)
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_ylim(0, 0.6)
        style_axes(ax)
    axes[0].set_ylabel("Churn rate")
    axes[0].text(2.45, overall + 0.012, f"overall {overall:.1%}", ha="right", color=MUTED, fontsize=9)
    fig.suptitle("Churn rate by segment (full dataset, 7,043 customers)", x=0.01, ha="left", color=INK, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_evaluation(y, curves, final_name, path):
    """curves: {label: proba}; first entry is the final model."""
    colors = [BLUE, ORANGE]
    fig, axes = new_fig(1, 3, figsize=(15, 4.6))
    ax_roc, ax_pr, ax_gain = axes
    for (label, p), c in zip(curves.items(), colors):
        fpr, tpr, _ = roc_curve(y, p)
        ax_roc.plot(fpr, tpr, color=c, linewidth=2, label=f"{label} (AUC {roc_auc_score(y, p):.3f})")
        prec, rec, _ = precision_recall_curve(y, p)
        ax_pr.plot(rec, prec, color=c, linewidth=2, label=f"{label} (AP {average_precision_score(y, p):.3f})")
        order = np.argsort(-p)
        gain = np.concatenate([[0], np.cumsum(y[order]) / y.sum()])
        ax_gain.plot(np.linspace(0, 1, len(gain)), gain, color=c, linewidth=2, label=label)
    ax_roc.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1, label="Chance")
    ax_pr.axhline(y.mean(), color=MUTED, linestyle="--", linewidth=1, label=f"No skill ({y.mean():.2f})")
    ax_gain.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1, label="Random targeting")

    ax_roc.set(xlabel="False positive rate", ylabel="True positive rate")
    ax_pr.set(xlabel="Recall", ylabel="Precision")
    ax_gain.set(xlabel="Share of customers contacted (riskiest first)", ylabel="Share of churners reached")
    for ax, title in zip(axes, ["ROC curve", "Precision-recall curve", "Cumulative gain"]):
        style_axes(ax, grid_axis="both")
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower right" if ax is not ax_pr else "upper right")
    ax_pr.set_ylim(0, 1.02)
    ax_gain.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax_gain.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    fig.suptitle(f"Held-out test set: {final_name} vs baseline", x=0.01, ha="left", color=INK, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_confusion(y, p, thr, path):
    cm = confusion_matrix(y, (p >= thr).astype(int))
    fig, ax = new_fig(figsize=(4.8, 4.4))
    ax.imshow(cm, cmap=BLUE_RAMP)
    labels = [["True negative", "False positive"], ["False negative", "True positive"]]
    for i in range(2):
        for j in range(2):
            dark = cm[i, j] > cm.max() * 0.45
            ax.text(j, i - 0.08, f"{cm[i, j]:,}", ha="center", va="center", fontsize=16, color="#ffffff" if dark else INK)
            ax.text(j, i + 0.18, labels[i][j], ha="center", va="center", fontsize=9, color="#ffffff" if dark else INK2)
    ax.set_xticks([0, 1], ["Predicted stay", "Predicted churn"])
    ax.set_yticks([0, 1], ["Actually stayed", "Actually churned"])
    ax.tick_params(colors=INK2, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"Confusion matrix (threshold {thr:.2f})", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def plot_importance(imp_df, path):
    top = imp_df.head(10).iloc[::-1]
    fig, ax = new_fig(figsize=(7.5, 4.6))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color=BLUE, height=0.55,
            error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 0})
    ax.set_xlabel("Drop in test ROC-AUC when the column is shuffled")
    ax.set_title("Permutation importance (top 10 raw columns)", loc="left", color=INK, fontsize=11)
    style_axes(ax, grid_axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 5. Main
# ----------------------------------------------------------------------------
def main():
    RESULTS.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    X, y = load_data()
    print(f"Loaded {X.shape[0]:,} customers, {X.shape[1]} raw features, churn rate {y.mean():.1%}")
    plot_eda(X, y, RESULTS / "eda_churn_by_segment.png")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    y_tr_a, y_te_a = y_tr.to_numpy(), y_te.to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    print(f"Train {len(X_tr):,} / test {len(X_te):,} (stratified)\n")

    # --- Model comparison with default hyperparameters --------------------
    print("== 5-fold CV, default hyperparameters, engineered features ==")
    rows = [cv_row(name, build_pipeline(m, True), X_tr, y_tr_a, cv) for name, m in make_models().items()]
    cv_df = pd.DataFrame(rows)
    cv_df.to_csv(RESULTS / "cv_model_comparison.csv", index=False)
    print(cv_df[["model", "roc_auc_mean", "pr_auc_mean", "recall_mean", "f1_mean"]].round(3).to_string(index=False))

    # --- Ablation: raw features vs engineered features --------------------
    print("\n== Ablation: engineered features ==")
    abl = []
    for name, m in make_models().items():
        if name.startswith("Dummy"):
            continue
        raw = cv_row(name, build_pipeline(m, False), X_tr, y_tr_a, cv)
        eng = cv_df.loc[cv_df["model"] == name].iloc[0]
        abl.append(
            {
                "model": name,
                "roc_auc_raw": raw["roc_auc_mean"],
                "roc_auc_engineered": eng["roc_auc_mean"],
                "pr_auc_raw": raw["pr_auc_mean"],
                "pr_auc_engineered": eng["pr_auc_mean"],
            }
        )
    abl_df = pd.DataFrame(abl)
    abl_df.to_csv(RESULTS / "feature_ablation.csv", index=False)
    print(abl_df.round(4).to_string(index=False))

    # --- Hyperparameter tuning (train data only) --------------------------
    print("\n== Tuning (scoring = PR-AUC, 5-fold CV on train) ==")
    tuned, tune_rows = {}, []
    for name, m in make_models().items():
        if name.startswith("Dummy"):
            continue
        est, score, params = tune(name, m, X_tr, y_tr_a, cv)
        tuned[name] = (est, score)
        tune_rows.append({"model": name, "cv_pr_auc": score, "best_params": json.dumps(params)})
        print(f"{name}: CV PR-AUC {score:.4f}  {params}")
    pd.DataFrame(tune_rows).to_csv(RESULTS / "tuning_summary.csv", index=False)

    final_name = max(tuned, key=lambda k: tuned[k][1])
    final_est = tuned[final_name][0]
    print(f"\nSelected by CV: {final_name}")

    # --- Threshold from out-of-fold TRAIN predictions ----------------------
    oof = cross_val_predict(final_est, X_tr, y_tr_a, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    thr = best_f1_threshold(y_tr_a, oof)
    print(f"Decision threshold (max F1 on out-of-fold train predictions): {thr:.2f}")

    # --- Final evaluation on the held-out test set -------------------------
    p_final = final_est.predict_proba(X_te)[:, 1]
    baseline = build_pipeline(make_models()["Logistic Regression"], engineer=False).fit(X_tr, y_tr_a)
    p_base = baseline.predict_proba(X_te)[:, 1]
    dummy = build_pipeline(make_models()["Dummy (class prior)"], engineer=False).fit(X_tr, y_tr_a)
    p_dummy = dummy.predict_proba(X_te)[:, 1]

    def summary(p, t):
        d = {"roc_auc": float(roc_auc_score(y_te_a, p)), "pr_auc": float(average_precision_score(y_te_a, p))}
        d.update(threshold_metrics(y_te_a, p, t))
        return d

    test = {
        "dummy": summary(p_dummy, 0.5),
        "baseline_logreg_raw_features_t0.5": summary(p_base, 0.5),
        "final_t0.5": summary(p_final, 0.5),
        "final_tuned_threshold": summary(p_final, thr),
    }
    test["final_bootstrap"] = bootstrap_ci(y_te_a, p_final)
    test["final_capture_rate"] = {f"top_{int(f * 100)}pct": capture_rate(y_te_a, p_final, f) for f in (0.1, 0.2, 0.3)}
    test["final_model"] = final_name
    test["test_churn_rate"] = float(y_te_a.mean())
    print("\n== Test set ==")
    for k, v in test.items():
        print(k, json.dumps(v, indent=None))

    # --- Error analysis ---------------------------------------------------
    te_feat = add_features(X_te).reset_index(drop=True)
    seg = []
    for col in ("Contract", "tenure_bucket", "InternetService"):
        seg += segment_report(te_feat, y_te_a, p_final, thr, col)
    seg_df = pd.DataFrame(seg)
    seg_df.to_csv(RESULTS / "error_analysis_by_segment.csv", index=False)
    print("\n== Error analysis by segment (test, tuned threshold) ==")
    print(seg_df.round(3).to_string(index=False))

    # --- Permutation importance (raw columns) -----------------------------
    imp = permutation_importance(
        final_est, X_te, y_te_a, scoring="roc_auc", n_repeats=15, random_state=SEED, n_jobs=-1
    )
    imp_df = (
        pd.DataFrame({"feature": X_te.columns, "importance_mean": imp.importances_mean, "importance_std": imp.importances_std})
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    imp_df.to_csv(RESULTS / "permutation_importance.csv", index=False)
    print("\n== Permutation importance ==")
    print(imp_df.head(8).round(4).to_string(index=False))

    # --- Plots + artifacts -----------------------------------------------
    plot_evaluation(
        y_te_a, {f"Tuned {final_name}": p_final, "Baseline logistic regression": p_base}, final_name,
        RESULTS / "test_curves.png",
    )
    plot_confusion(y_te_a, p_final, thr, RESULTS / "confusion_matrix.png")
    plot_importance(imp_df, RESULTS / "permutation_importance.png")

    pd.DataFrame({"y_true": y_te_a, "churn_probability": p_final, "predicted": (p_final >= thr).astype(int)}).to_csv(
        RESULTS / "test_predictions.csv", index=False
    )
    (RESULTS / "metrics.json").write_text(json.dumps(test, indent=2))
    joblib.dump({"pipeline": final_est, "threshold": thr, "model_name": final_name}, MODELS_DIR / "churn_model.joblib")
    print("\nSaved results/ and models/churn_model.joblib")


if __name__ == "__main__":
    main()
