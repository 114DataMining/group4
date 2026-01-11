import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
from sklearn.model_selection import (train_test_split,StratifiedKFold,GridSearchCV)  
from sklearn.tree import DecisionTreeClassifier, plot_tree 
from sklearn.metrics import ( 
accuracy_score, f1_score, roc_auc_score, 
roc_curve, auc,precision_score, recall_score 
)
df_onehot = pd.read_csv("C:\\Users\\arnie\\OneDrive\\桌面\\資料探勘\\vgsales_focus_onehot.csv")
X = df_onehot.drop(columns=["hit"]).copy()
y = df_onehot["hit"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,      # ✅ 30% 當測試集 → 7:3
    random_state=42,
    stratify=y          # ✅ 保持 0/1 比例一致（分類很重要）
)

print("X_train:", X_train.shape, "X_test:", X_test.shape)
print("y_train hit rate:", y_train.mean(), "y_test hit rate:", y_test.mean())
RANDOM_STATE = 42 
N_SPLITS = 5 
TARGET_COL = "hit" 
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, 
random_state=RANDOM_STATE) 
base_model = DecisionTreeClassifier( 
random_state=RANDOM_STATE, 
class_weight="balanced" 
) 
param_grid = { 
"max_depth": [2, 3, 4, 5, 6, 8, 10, None], 
"min_samples_split": [2, 5, 10, 20], 
"min_samples_leaf": [1, 2, 5, 10], 
"criterion": ["gini", "entropy"] 
} 
grid = GridSearchCV( 
estimator=base_model, 
param_grid=param_grid, 
scoring="roc_auc", 
cv=cv, 
n_jobs=-1 ) 

grid.fit(X_train, y_train) 
best_params = grid.best_params_ 
best_model = grid.best_estimator_ 
import pandas as pd

# 取得 GridSearchCV 所有結果
results = pd.DataFrame(grid.cv_results_)

# 只留下我們關心的欄位
auc_table = results[[
    "param_max_depth",
    "param_min_samples_split",
    "param_min_samples_leaf",
    "param_criterion",
    "mean_test_score",
    "std_test_score"
]].sort_values(by="mean_test_score", ascending=False)

# 重新命名欄位
auc_table = auc_table.rename(columns={
    "param_max_depth": "max_depth",
    "param_min_samples_split": "min_samples_split",
    "param_min_samples_leaf": "min_samples_leaf",
    "param_criterion": "criterion",
    "mean_test_score": "Mean_AUC",
    "std_test_score": "Std_AUC"
})

# 顯示前10名（或全部）
print(auc_table.head(10))


print("\n=== Converged (Selected) Hyperparameters ===") 
print("Best params:", best_params) 
print("Best CV AUC:", grid.best_score_) 

model_cv = DecisionTreeClassifier( 
random_state=RANDOM_STATE, 
class_weight="balanced", 
**best_params 
) 
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,random_state=RANDOM_STATE) 
accs, f1s, aucs = [], [], [] 
print("\n=== 5-fold results (per fold) ===") 
for fold, (tr_idx, te_idx) in enumerate(skf.split(X_train, y_train), start=1): 
    X_tr, X_te = X_train.iloc[tr_idx], X_train.iloc[te_idx] 
    y_tr, y_te = y_train.iloc[tr_idx], y_train.iloc[te_idx] 
    n_pos_fold = int((y_tr == 1).sum()) 
    n_neg_fold = int((y_tr == 0).sum()) 
    pos_rate_fold = n_pos_fold / (n_pos_fold + n_neg_fold) 
    model_cv.fit(X_tr, y_tr) 
    y_pred = model_cv.predict(X_te) 
    y_score = model_cv.predict_proba(X_te)[:, 1] 
    acc = accuracy_score(y_te, y_pred) 
    f1 = f1_score(y_te, y_pred, zero_division=0) 
    auc = roc_auc_score(y_te, y_score) 
    accs.append(acc) 
    f1s.append(f1) 
    aucs.append(auc) 
    print(f"Fold {fold}: pos_rate(train)={pos_rate_fold:.3f} | " 
    f"ACC={acc:.4f} | F1={f1:.4f} | AUC={auc:.4f}") 
print("\n=== 5-fold summary (mean ± std) ===") 
print(f"ACC: {np.mean(accs):.4f} ± {np.std(accs, ddof=1):.4f}") 
print(f"F1 : {np.mean(f1s):.4f} ± {np.std(f1s, ddof=1):.4f}") 
print(f"AUC: {np.mean(aucs):.4f} ± {np.std(aucs, ddof=1):.4f}")
from sklearn.metrics import roc_curve, auc
mean_fpr = np.linspace(0, 1, 100)
tprs = []
aucs = []

plt.figure(figsize=(8, 6))

for fold, (tr_idx, te_idx) in enumerate(skf.split(X_train, y_train), start=1):
    X_tr, X_te = X_train.iloc[tr_idx], X_train.iloc[te_idx]
    y_tr, y_te = y_train.iloc[tr_idx], y_train.iloc[te_idx]

    model_cv.fit(X_tr, y_tr)
    y_score = model_cv.predict_proba(X_te)[:, 1]

    fpr, tpr, _ = roc_curve(y_te, y_score)
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)

    # 插值到共同的 FPR
    tpr_interp = np.interp(mean_fpr, fpr, tpr)
    tpr_interp[0] = 0.0
    tprs.append(tpr_interp)

    plt.plot(fpr, tpr, lw=1, alpha=0.4,
             label=f'Fold {fold} ROC (AUC = {roc_auc:.3f})')

# 平均 ROC
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs, ddof=1)

plt.plot(mean_fpr, mean_tpr, color='black',
         label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})',
         lw=2)

# 隨機分類線
plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='gray')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('5-Fold Cross-Validation ROC Curves')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()


model = DecisionTreeClassifier( 
random_state=RANDOM_STATE, 
class_weight="balanced",   
max_depth=5, 
min_samples_leaf=5, 
min_samples_split=2, 
criterion="gini" 
) 
model.fit(X_train, y_train) 
 

y_pred = model.predict(X_test)
y_score = model.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc_roc = roc_auc_score(y_test, y_score)

print("\n=== Final Test Performance (Fixed Hyperparams) ===")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1       : {f1:.4f}")
print(f"AUC      : {auc_roc:.4f}") 
# ============================================================ 
# 5) 產出決策樹圖
# ============================================================ 
tree_path = os.path.join("C:\\Users\\arnie\\OneDrive\\桌面\\資料探勘\\group4", "decision_tree_fixed.png") 
plt.figure(figsize=(22, 10)) 
plot_tree( 
model, 
feature_names=X_train.columns, 
class_names=["0", "1"],     
filled=True, 
rounded=True, 
max_depth=5,                
fontsize=9 
) 
plt.title("Decision Tree (Fixed Hyperparams) - Top levels") 
plt.tight_layout() 
plt.savefig(tree_path, dpi=300) 
plt.close() 
print(f"Saved decision tree figure: {tree_path}") 
# 6) 畫 ROC curve + 算 AUC
from sklearn.metrics import roc_auc_score, roc_curve, auc
 
roc_path = os.path.join("C:\\Users\\arnie\\OneDrive\\桌面\\資料探勘\\group4", "roc_curve_fixed_test.png") 
fpr, tpr, thresholds = roc_curve(y_test, y_score) 
roc_auc = auc(fpr, tpr) 
plt.figure(figsize=(7, 6)) 
plt.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})") 
plt.plot([0, 1], [0, 1], lw=1, linestyle="--", label="Random guess") 
plt.xlim([0.0, 1.0]) 
plt.ylim([0.0, 1.05]) 
plt.xlabel("False Positive Rate") 
plt.ylabel("True Positive Rate") 
plt.title("ROC Curve (Decision Tree, Final Test Set)") 
plt.legend(loc="lower right") 
plt.tight_layout() 
plt.savefig(roc_path, dpi=300) 
plt.close() 
print(f"Saved ROC curve figure: {roc_path}") 
print(f"ROC AUC (recomputed via curve): {roc_auc:.4f}") 



importances = best_model.feature_importances_
features = X_train.columns

df_imp = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

plt.figure()
plt.barh(df_imp["Feature"], df_imp["Importance"])
plt.gca().invert_yaxis()
plt.title("Decision Tree Feature Importance")
plt.show()
