"""
Breast Cancer Diagnosis Prediction Using Machine Learning
Dataset : Breast Cancer Wisconsin (Diagnostic) - UCI ML Repository
Author  : Swasthika Battcharya | Roll No: 25RBAID0002
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    classification_report,
)

warnings.filterwarnings("ignore")
os.makedirs("figs", exist_ok=True)
RS = 42

# 1. LOAD DATASET
data = load_breast_cancer(as_frame=True)
df = data.frame.copy()
df["diagnosis"] = df["target"].map({0: "Malignant", 1: "Benign"})
print("Shape:", df.shape)
print(df["diagnosis"].value_counts())
print("Missing values:", int(df.isnull().sum().sum()))
print("Duplicate rows:", int(df.duplicated().sum()))

# 2. EDA
X = data.data
y = data.target
fig, ax = plt.subplots(figsize=(5, 4))
counts = df["diagnosis"].value_counts()
ax.bar(counts.index, counts.values, color=["#2E75B6", "#C0504D"])
for i, v in enumerate(counts.values):
    ax.text(i, v + 5, str(v), ha="center")
ax.set_title("Class Distribution")
ax.set_ylabel("Number of patients")
plt.tight_layout()
plt.savefig("figs/class_distribution.png", dpi=150)
plt.close()

fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for ax, col in zip(
    axes.ravel(),
    [
        "mean radius",
        "mean texture",
        "mean perimeter",
        "mean area",
        "mean concavity",
        "mean smoothness",
    ],
):
    for lab, c in [("Malignant", "#C0504D"), ("Benign", "#2E75B6")]:
        ax.hist(df[df.diagnosis == lab][col], bins=25, alpha=0.6, label=lab, color=c)
    ax.set_title(col)
    ax.legend()
plt.tight_layout()
plt.savefig("figs/histograms.png", dpi=150)
plt.close()

fig, axes = plt.subplots(1, 4, figsize=(13, 4))
for ax, col in zip(
    axes, ["mean radius", "mean area", "mean concavity", "mean smoothness"]
):
    ax.boxplot(
        [df[df.diagnosis == "Malignant"][col], df[df.diagnosis == "Benign"][col]],
        tick_labels=["Malignant", "Benign"],
    )
    ax.set_title(col)
plt.tight_layout()
plt.savefig("figs/boxplots.png", dpi=150)
plt.close()

mean_cols = [c for c in X.columns if c.startswith("mean")]
corr = X[mean_cols].corr()
fig, ax = plt.subplots(figsize=(8, 6.5))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(mean_cols)))
ax.set_yticks(range(len(mean_cols)))
ax.set_xticklabels([c.replace("mean ", "") for c in mean_cols], rotation=90)
ax.set_yticklabels([c.replace("mean ", "") for c in mean_cols])
plt.colorbar(im)
ax.set_title("Correlation Heatmap (Mean Features)")
plt.tight_layout()
plt.savefig("figs/correlation_heatmap.png", dpi=150)
plt.close()

target_corr = X.corrwith(y).sort_values()
top = pd.concat([target_corr.head(8), target_corr.tail(4)])
print("\nTop correlations with target (0=malignant,1=benign):\n", top.round(3))
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(
    top.index, top.values, color=["#C0504D" if v < 0 else "#2E75B6" for v in top.values]
)
ax.set_title("Feature Correlation with Diagnosis (Benign = 1)")
plt.tight_layout()
plt.savefig("figs/target_correlation.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(6, 5))
for lab, c in [("Malignant", "#C0504D"), ("Benign", "#2E75B6")]:
    s = df[df.diagnosis == lab]
    ax.scatter(
        s["mean radius"], s["mean concave points"], alpha=0.6, c=c, label=lab, s=18
    )
ax.set_xlabel("mean radius")
ax.set_ylabel("mean concave points")
ax.set_title("Mean Radius vs Mean Concave Points")
ax.legend()
plt.tight_layout()
plt.savefig("figs/scatter.png", dpi=150)
plt.close()

# 3. SPLIT AND SCALE
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RS, stratify=y
)
print("\nTrain:", X_train.shape, "Test:", X_test.shape)
print(
    "Train class counts:",
    np.bincount(y_train),
    "Test class counts:",
    np.bincount(y_test),
)
scaler = StandardScaler().fit(X_train)
Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)

# 4. MODELS
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RS),
    "Decision Tree": DecisionTreeClassifier(random_state=RS),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RS),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=RS),
    "Gaussian Naive Bayes": GaussianNB(),
}
rows, fitted, probs, cms = [], {}, {}, {}
for name, m in models.items():
    m.fit(Xtr, y_train)
    fitted[name] = m
    p = m.predict(Xte)
    pr = m.predict_proba(Xte)[:, 1]
    probs[name] = pr
    cms[name] = confusion_matrix(y_test, p)
    rows.append(
        {
            "Model": name,
            "Accuracy": accuracy_score(y_test, p),
            "Precision": precision_score(y_test, p),
            "Recall": recall_score(y_test, p),
            "F1": f1_score(y_test, p),
            "ROC-AUC": roc_auc_score(y_test, pr),
        }
    )
res = pd.DataFrame(rows).set_index("Model")
print("\nTEST RESULTS\n", res.round(4))
for n in models:
    print("\n", n, "confusion matrix [[TN FP][FN TP]]:", cms[n].tolist())
print(
    "\nClassification report (SVM):\n",
    classification_report(
        y_test, fitted["SVM (RBF)"].predict(Xte), target_names=["Malignant", "Benign"]
    ),
)

# Model comparison chart
fig, ax = plt.subplots(figsize=(10, 5))
w = 0.16
xs = np.arange(len(res))
for i, met in enumerate(["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]):
    ax.bar(xs + (i - 2) * w, res[met], w, label=met)
ax.set_xticks(xs)
ax.set_xticklabels(res.index, rotation=20)
ax.set_ylim(0.85, 1.01)
ax.set_title("Model Comparison on Test Set")
ax.legend(ncol=5, loc="lower center")
plt.tight_layout()
plt.savefig("figs/model_comparison.png", dpi=150)
plt.close()

# Confusion matrices
fig, axes = plt.subplots(2, 3, figsize=(11, 7))
for ax, (n, cm) in zip(axes.ravel(), cms.items()):
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                fontsize=14,
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Malignant", "Benign"])
    ax.set_yticklabels(["Malignant", "Benign"])
    ax.set_title(n)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("figs/confusion_matrices.png", dpi=150)
plt.close()

# ROC curves
fig, ax = plt.subplots(figsize=(6.5, 5.5))
for n, pr in probs.items():
    fpr, tpr, _ = roc_curve(y_test, pr)
    ax.plot(fpr, tpr, label=f"{n} (AUC={roc_auc_score(y_test, pr):.3f})")
ax.plot([0, 1], [0, 1], "k--")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves")
ax.legend(fontsize=8, loc="lower right")
plt.tight_layout()
plt.savefig("figs/roc_curves.png", dpi=150)
plt.close()

# 5. CROSS VALIDATION
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
cv_rows = []
for name, m in models.items():
    pipe = Pipeline([("sc", StandardScaler()), ("m", m)])
    sc = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy")
    cv_rows.append(
        {
            "Model": name,
            "Mean": sc.mean(),
            "Std": sc.std(),
            "Folds": sc.round(4).tolist(),
        }
    )
cv = pd.DataFrame(cv_rows).set_index("Model")
print("\n5-FOLD CV\n", cv[["Mean", "Std"]].round(4))
for n, r in cv.iterrows():
    print(n, r["Folds"])
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(cv.index, cv["Mean"], yerr=cv["Std"], capsize=4, color="#2E75B6")
ax.set_ylim(0.88, 1.0)
ax.set_title("5-Fold Cross-Validation Accuracy")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("figs/cv_accuracy.png", dpi=150)
plt.close()

# 6. KNN effect of k
ks = list(range(1, 26))
kacc = []
for k in ks:
    kacc.append(
        accuracy_score(
            y_test, KNeighborsClassifier(n_neighbors=k).fit(Xtr, y_train).predict(Xte)
        )
    )
print("\nKNN accuracy by k:", {k: round(float(a), 4) for k, a in zip(ks, kacc)})
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(ks, kacc, marker="o")
ax.set_xlabel("k")
ax.set_ylabel("Test accuracy")
ax.set_title("Effect of k on KNN Accuracy")
plt.tight_layout()
plt.savefig("figs/knn_k.png", dpi=150)
plt.close()

# 7. DECISION TREE DEPTH (over/underfitting)
depths = list(range(1, 11))
tra, tea = [], []
for d in depths:
    t = DecisionTreeClassifier(max_depth=d, random_state=RS).fit(Xtr, y_train)
    tra.append(t.score(Xtr, y_train))
    tea.append(t.score(Xte, y_test))
print("\nDepth train acc:", np.round(tra, 4).tolist())
print("Depth test acc:", np.round(tea, 4).tolist())
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(depths, tra, marker="o", label="Train")
ax.plot(depths, tea, marker="s", label="Test")
ax.set_xlabel("max_depth")
ax.set_ylabel("Accuracy")
ax.set_title("Decision Tree Depth vs Accuracy")
ax.legend()
plt.tight_layout()
plt.savefig("figs/tree_depth.png", dpi=150)
plt.close()

# 8. FEATURE IMPORTANCE
rf = fitted["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 10 RF importances:\n", imp.head(10).round(4))
fig, ax = plt.subplots(figsize=(7, 5))
imp.head(10)[::-1].plot(kind="barh", ax=ax, color="#2E75B6")
ax.set_title("Top 10 Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig("figs/feature_importance.png", dpi=150)
plt.close()

# 9. PCA
pca = PCA().fit(StandardScaler().fit_transform(X))
cum = np.cumsum(pca.explained_variance_ratio_)
print(
    "\nPCA explained variance (first 5):",
    pca.explained_variance_ratio_[:5].round(4).tolist(),
)
print("Components for 95% variance:", int(np.argmax(cum >= 0.95) + 1))
Z = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(X))
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].plot(range(1, 31), cum, marker="o", ms=3)
axes[0].axhline(0.95, ls="--", c="r")
axes[0].set_xlabel("Components")
axes[0].set_ylabel("Cumulative variance")
axes[0].set_title("PCA Explained Variance")
for lab, v, c in [("Malignant", 0, "#C0504D"), ("Benign", 1, "#2E75B6")]:
    axes[1].scatter(Z[y == v, 0], Z[y == v, 1], s=14, alpha=0.6, c=c, label=lab)
axes[1].set_title("PCA 2-D Projection")
axes[1].legend()
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")
plt.tight_layout()
plt.savefig("figs/pca.png", dpi=150)
plt.close()

# 10. HYPERPARAMETER TUNING (SVM)
grid = GridSearchCV(
    SVC(probability=True, random_state=RS),
    {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", 0.01, 0.001],
        "kernel": ["rbf", "linear"],
    },
    cv=5,
    scoring="accuracy",
)
grid.fit(Xtr, y_train)
best = grid.best_estimator_
pb = best.predict(Xte)
print("\nBest SVM params:", grid.best_params_, "CV acc:", round(grid.best_score_, 4))
print(
    "Tuned SVM test acc:",
    round(accuracy_score(y_test, pb), 4),
    "Recall:",
    round(recall_score(y_test, pb), 4),
    "Precision:",
    round(precision_score(y_test, pb), 4),
    "F1:",
    round(f1_score(y_test, pb), 4),
)
print("Tuned SVM confusion matrix:", confusion_matrix(y_test, pb).tolist())

# 11. NEW PATIENT PREDICTION (first 3 test rows use real samples)
final = best
sample = X_test.iloc[[0, 1, 2]]
pp = final.predict(scaler.transform(sample))
ppr = final.predict_proba(scaler.transform(sample))
for i in range(3):
    print(
        f"Sample {i+1}: actual={'Benign' if y_test.iloc[i]==1 else 'Malignant'} "
        f"predicted={'Benign' if pp[i]==1 else 'Malignant'} "
        f"P(malignant)={ppr[i,0]:.3f} P(benign)={ppr[i,1]:.3f}"
    )
print(
    "\nDescriptive stats (mean features):\n",
    X[mean_cols].describe().T[["mean", "std", "min", "max"]].round(3),
)
