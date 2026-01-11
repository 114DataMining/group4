# ============================================================
# 8) Baseline Logistic Regression（Pipeline）
# ============================================================
lr_baseline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000, random_state=42))
])

lr_baseline.fit(X_train, y_train)

y_pred_base = lr_baseline.predict(X_test)
y_proba_base = lr_baseline.predict_proba(X_test)[:, 1]

acc_base = accuracy_score(y_test, y_pred_base)
auc_base = roc_auc_score(y_test, y_proba_base)

print("\n=== Baseline Logistic (Test) ===")
print(f"Accuracy: {acc_base:.4f}")
print(f"AUC     : {auc_base:.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_base))
print("\nClassification Report:\n", classification_report(y_test, y_pred_base, digits=4))

# ROC (baseline)
fpr, tpr, _ = roc_curve(y_test, y_proba_base)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"Baseline LR (AUC={auc_base:.3f})")
plt.plot([0,1],[0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Baseline Logistic Regression")
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# 9) 5-Fold Cross Validation（在訓練集上看穩定性）
# ============================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_validate(
    lr_baseline, X_train, y_train,
    cv=cv,
    scoring={"acc":"accuracy", "auc":"roc_auc"},
    return_train_score=False
)

print("\n=== 5-Fold CV on Train (Baseline LR) ===")
print(f"CV Accuracy: {cv_scores['test_acc'].mean():.4f} ± {cv_scores['test_acc'].std():.4f}")
print(f"CV AUC     : {cv_scores['test_auc'].mean():.4f} ± {cv_scores['test_auc'].std():.4f}")

# ============================================================
# 10) 超參數調整：GridSearchCV（只用訓練集做）
# ============================================================
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=5000, random_state=42))
])

param_grid = [
    {
        "clf__solver": ["liblinear"],
        "clf__penalty": ["l1", "l2"],
        "clf__C": [0.01, 0.1, 1, 10, 100],
        "clf__class_weight": [None, "balanced"]
    },
    {
        "clf__solver": ["lbfgs"],
        "clf__penalty": ["l2"],
        "clf__C": [0.01, 0.1, 1, 10, 100],
        "clf__class_weight": [None, "balanced"]
    }
]

grid = GridSearchCV(
    pipe,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("\n=== GridSearchCV (Train only) ===")
print("Best CV AUC:", grid.best_score_)
print("Best Params:", grid.best_params_)

# ============================================================
# 11) Tuned Logistic：用最佳模型做 Test 評估 + ROC
# ============================================================
best_lr = grid.best_estimator_

y_pred_tuned = best_lr.predict(X_test)
y_proba_tuned = best_lr.predict_proba(X_test)[:, 1]

acc_tuned = accuracy_score(y_test, y_pred_tuned)
auc_tuned = roc_auc_score(y_test, y_proba_tuned)

print("\n=== Tuned Logistic (Test) ===")
print(f"Accuracy: {acc_tuned:.4f}")
print(f"AUC     : {auc_tuned:.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_tuned))
print("\nClassification Report:\n", classification_report(y_test, y_pred_tuned, digits=4))

# ROC (tuned)
fpr2, tpr2, _ = roc_curve(y_test, y_proba_tuned)
plt.figure(figsize=(6,5))
plt.plot(fpr2, tpr2, label=f"Tuned LR (AUC={auc_tuned:.3f})")
plt.plot([0,1],[0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Tuned Logistic Regression")
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# 12) Baseline vs Tuned（同一張 ROC 圖比較）
# ============================================================
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"Baseline LR (AUC={auc_base:.3f})")
plt.plot(fpr2, tpr2, label=f"Tuned LR (AUC={auc_tuned:.3f})")
plt.plot([0,1],[0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Comparison - Baseline vs Tuned")
plt.legend()
plt.tight_layout()
plt.show()

print("\n=== Summary (Test) ===")
print(f"Baseline: Acc={acc_base:.4f}, AUC={auc_base:.4f}")
print(f"Tuned Accuracy : {acc_tuned:.4f}")
print(f"Tuned AUC      : {auc_tuned:.4f}")


# ============================================================
# 13) 係數解釋（用最佳模型 best_lr）
# ============================================================
coef = best_lr.named_steps["clf"].coef_[0]
feature_names = X_train.columns

coef_df = (
    pd.DataFrame({
        "feature": feature_names,
        "coef": coef,
        "odds_ratio": np.exp(coef)
    })
    .sort_values("coef", ascending=False)
)

print("\n=== Top Positive Coefficients ===")
print(coef_df.head(10))

print("\n=== Top Negative Coefficients ===")
print(coef_df.tail(10))

# 係數圖（取前後各 8 個）
top_plot = pd.concat([coef_df.head(8), coef_df.tail(8)])

plt.figure(figsize=(9,5))
plt.barh(top_plot["feature"], top_plot["coef"])
plt.axvline(0, color="black", linewidth=1)
plt.title("Tuned Logistic Coefficients (Top +/-)")
plt.xlabel("Coefficient")
plt.tight_layout()
plt.show()