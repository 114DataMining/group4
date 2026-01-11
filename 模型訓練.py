#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 0) 讀取資料 + 基本清理
# ============================================================
df = pd.read_csv("vgsales.csv")
df.columns = df.columns.str.strip().str.lower()

# 把欄位值也清乾淨（避免 "Wii " 之類導致篩不到）
df["platform"]  = df["platform"].astype(str).str.strip().str.upper()
df["genre"]     = df["genre"].astype(str).str.strip()
df["publisher"] = df["publisher"].astype(str).str.strip()

# 丟掉必要欄位缺值
df = df.dropna(subset=["platform", "genre", "publisher", "year", "global_sales"]).copy()
df["year"] = df["year"].astype(int)

print("原始資料 shape:", df.shape)
print("平台 unique 前20:", df["platform"].unique()[:20])

# ============================================================
# 1) 平台固定：WII / PS3 / X360 / PC
# ============================================================
keep_platforms = ["WII", "PS3", "X360", "PC"]
df_p = df[df["platform"].isin(keep_platforms)].copy()

print("\n只留指定平台後 shape:", df_p.shape)
print("平台分布：\n", df_p["platform"].value_counts())

if df_p.empty:
    raise ValueError("篩選後資料為空：請檢查 df['platform'].unique() 的平台寫法是否不同。")

# ============================================================
# 2) PC 同世代年份限制
#    同世代年份 = Wii/PS3/X360 在資料中出現的 min~max year
# ============================================================
base_platforms = ["WII", "PS3", "X360"]
base = df_p[df_p["platform"].isin(base_platforms)].copy()

min_year = int(base["year"].min())
max_year = int(base["year"].max())

print(f"\n同世代年份範圍（由 Wii/PS3/X360 決定）: {min_year} ~ {max_year}")

df_p = df_p[df_p["year"].between(min_year, max_year)].copy()

print("套用同世代年份後 shape:", df_p.shape)
print("平台分布（含PC同世代限制後）：\n", df_p["platform"].value_counts())

if df_p.empty:
    raise ValueError("套用同世代年份後資料為空：代表該年份範圍內資料不足。")

# ============================================================
# 3) 目標變數 HIT：用 global_sales 中位數（只在這個子資料上算）
# ============================================================
median_sales = df_p["global_sales"].median()
df_p["hit"] = (df_p["global_sales"] >= median_sales).astype(int)

print(f"\nGlobal_Sales 中位數 = {median_sales:.3f} 百萬套")
print("HIT 比例 =", df_p["hit"].mean())

# ============================================================
# 4) Top10 Publisher（二元特徵）
#    用 hit 的總數挑 Top10（比較符合「暢銷」概念）
# ============================================================
top_publishers = (
    df_p.groupby("publisher")["hit"].sum()
    .sort_values(ascending=False)
    .head(10)
    .index
)
df_p["top10_publisher"] = df_p["publisher"].isin(top_publishers).astype(int)

print("\nTop10 Publisher：", list(top_publishers))

# ============================================================
# 5) 類型：Top5 + Others（不刪資料）
# ============================================================
top5_genres = df_p["genre"].value_counts().head(5).index.tolist()
df_p["genre_group"] = df_p["genre"].where(df_p["genre"].isin(top5_genres), "Others")

print("\nTop5 Genres:", top5_genres)
print("genre_group 分布：\n", df_p["genre_group"].value_counts())

# ============================================================
# 6) 最終資料（不要 year / 不要 era）
# ============================================================
df_final = df_p[["platform", "genre_group", "top10_publisher", "hit"]].copy()

print("\n最終資料 shape:", df_final.shape)
print("最終欄位：", df_final.columns.tolist())

# ============================================================
# 7) Hit 分布（給老師看的：目標變數長什麼樣）
# ============================================================
hit_counts = df_final["hit"].value_counts().sort_index()
plt.figure(figsize=(4,3))
plt.bar(["0 (Below Median)", "1 (>= Median)"], [hit_counts.get(0,0), hit_counts.get(1,0)])
plt.title("Target Distribution: HIT")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# ============================================================
# 8) 視覺化：平台 hit rate、類型 hit rate、Top10Publisher hit rate
# ============================================================
def plot_hit_rate(df_in, col, title):
    tbl = df_in.groupby(col)["hit"].mean().sort_values(ascending=False)
    plt.figure(figsize=(7,4))
    plt.bar(tbl.index.astype(str), tbl.values)
    plt.ylim(0, 1)
    plt.ylabel("Hit Rate")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
    return tbl

plot_hit_rate(df_final, "platform", "Hit Rate by Platform (Same Generation)")
plot_hit_rate(df_final, "genre_group", "Hit Rate by Genre (Top5 + Others)")

pub_tbl = df_final.groupby("top10_publisher")["hit"].mean()
plt.figure(figsize=(4,3))
plt.bar(["Non-Top10", "Top10"], [pub_tbl.get(0, np.nan), pub_tbl.get(1, np.nan)])
plt.ylim(0,1)
plt.title("Hit Rate: Top10 Publisher vs Others")
plt.tight_layout()
plt.show()

# ============================================================
# 9) 熱力圖：平台 × 類型 的 Hit Rate（重點：values=hit）
# ============================================================
pivot = df_final.pivot_table(
    index="platform",
    columns="genre_group",
    values="hit",
    aggfunc="mean"
).reindex(index=["WII","PS3","X360","PC"])  # 固定順序比較好講

plt.figure(figsize=(8,5))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="coolwarm", vmin=0, vmax=1)
plt.title("Hit Rate Heatmap: Platform vs Genre (Top5 + Others)")
plt.tight_layout()
plt.show()

# ============================================================
# 10) One-Hot Encoding + 輸出 CSV（給老師講解/丟模型用）
#    不用 drop_first：比較好講解（不會少一個類別）
# ============================================================
X = df_final[["platform", "genre_group", "top10_publisher"]].copy()
y = df_final["hit"].copy()

X_encoded = pd.get_dummies(X, columns=["platform", "genre_group"], drop_first=False)
X_encoded = X_encoded.astype(int)
df_onehot = X_encoded.copy()
df_onehot["hit"] = y.values

# 輸出檔案
df_final.to_csv("vgsales_focus_final.csv", index=False)         # 未編碼、可解釋
df_onehot.to_csv("vgsales_focus_onehot.csv", index=False)       # One-Hot 後、可丟模型

print("\n✅ 已輸出：vgsales_focus_final.csv（未編碼、可解釋）")
print("✅ 已輸出：vgsales_focus_onehot.csv（One-Hot 後、可丟模型）")
print("One-Hot shape:", df_onehot.shape)
print("One-Hot 欄位前25個：", df_onehot.columns[:25].tolist())

# ============================================================
# 11) 相關性熱力圖（分兩張）：Hit vs 平台(+Top10)、Hit vs 類型(+Top10)
#    這是你想要的「特徵與暢銷(Hit)的相關性熱力圖」感覺
# ============================================================
# 準備 One-Hot（不含 hit 先做）
df_corr = df_onehot.copy()

# 取平台dummy欄位
platform_cols = [c for c in df_corr.columns if c.startswith("platform_")]
genre_cols    = [c for c in df_corr.columns if c.startswith("genre_group_")]

def corr_heatmap_with_numbers(cols, title, figsize=(8,6)):
    sub = df_corr[["hit", "top10_publisher"] + cols].corr()

    plt.figure(figsize=figsize)
    im = plt.imshow(sub, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im)

    plt.xticks(range(len(sub.columns)), sub.columns, rotation=45, ha="right")
    plt.yticks(range(len(sub.index)), sub.index)

    for i in range(sub.shape[0]):
        for j in range(sub.shape[1]):
            v = sub.iloc[i, j]
            plt.text(
                j, i, f"{v:.2f}",
                ha="center", va="center",
                color="white" if abs(v) > 0.5 else "black",
                fontsize=8
            )

    plt.title(title)
    plt.tight_layout()
    plt.show()

# Hit vs Platform (+Top10Publisher)
corr_heatmap_with_numbers(
    platform_cols,
    "Correlation Heatmap: Hit vs Platform (+ Top10Publisher)",
    figsize=(8,6)
)

# Hit vs Genre (+Top10Publisher)
corr_heatmap_with_numbers(
    genre_cols,
    "Correlation Heatmap: Hit vs Genre (+ Top10Publisher)",
    figsize=(9,6)
)
from sklearn.model_selection import train_test_split
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
)




import matplotlib.pyplot as plt
import numpy as np
# ============================================================
# 1) 定義羅吉斯回歸模型
# ============================================================
lr_model = Pipeline([
    ("scaler", StandardScaler()),      # 特徵標準化
    ("clf", LogisticRegression(max_iter=2000, random_state=42))
])

# ============================================================
# 2) 5-Fold 交叉驗證（只在訓練集 X_train, y_train 上做）
# ============================================================
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# 定義模型
lr_model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000, random_state=42))
])

# 5-Fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("===== 5-Fold Cross Validation (per fold) =====")

fold_idx = 1
acc_list = []
auc_list = []

for train_idx, val_idx in cv.split(X_train, y_train):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    n_tr_total = len(y_tr)
    n_tr_0 = np.sum(y_tr == 0)
    n_tr_1 = np.sum(y_tr == 1)
    r_tr0 = n_tr_0 / n_tr_total
    r_tr1 = n_tr_1 / n_tr_total

    n_val_total = len(y_val)
    n_val_0 = np.sum(y_val == 0)
    n_val_1 = np.sum(y_val == 1)
    r_val0 = n_val_0 / n_val_total
    r_val1 = n_val_1 / n_val_total

    print(
        f"Fold {fold_idx}:\n"
        f"  訓練集 -> 0: {n_tr_0} ({r_tr0:.4f}), 1: {n_tr_1} ({r_tr1:.4f})\n"
        f"  驗證集 -> 0: {n_val_0} ({r_val0:.4f}), 1: {n_val_1} ({r_val1:.4f})"
    )

    lr_model.fit(X_tr, y_tr)
    y_proba = lr_model.predict_proba(X_val)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y_val, y_pred)
    auc_score = roc_auc_score(y_val, y_proba)

    acc_list.append(acc)
    auc_list.append(auc_score)

    print(f"Fold {fold_idx}: Accuracy={acc:.4f}")
    fold_idx += 1

print(f"\n平均 CV Accuracy: {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")
print(f"平均 CV ROC-AUC : {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}")



# ============================================================
# 3) Fit 模型在整個訓練集
# ============================================================
lr_model.fit(X_train, y_train)

# ============================================================
# 4) 測試集評估
# ============================================================
y_proba = lr_model.predict_proba(X_test)[:, 1]  # 預測為 1 的機率
y_pred = (y_proba >= 0.5).astype(int)          # 閾值 0.5

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
print("\n===== Test Set Evaluation =====")
print(f"Test Accuracy: {acc:.4f}")
print(f"Test ROC-AUC : {auc:.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 計算混淆矩陣
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.xticks([0.5,1.5], ['Non-Hit (0)','Hit (1)'])
plt.yticks([0.5,1.5], ['Non-Hit (0)','Hit (1)'], rotation=0)
plt.tight_layout()
plt.show()


# ============================================================
# 5) 畫 ROC Curve
# ============================================================
from sklearn.metrics import roc_curve, auc

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

plt.figure(figsize=(7,6))

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fold_idx = 1
for train_idx, val_idx in cv.split(X_train, y_train):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    lr_model.fit(X_tr, y_tr)
    y_proba = lr_model.predict_proba(X_val)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, y_proba)
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)

    tpr_interp = np.interp(mean_fpr, fpr, tpr)
    tpr_interp[0] = 0.0
    tprs.append(tpr_interp)
    plt.plot(fpr, tpr, lw=1, alpha=0.7, label=f"Fold {fold_idx} (AUC={roc_auc:.3f})")
    fold_idx += 1

# 平均 ROC
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color="black", lw=2, linestyle="--",
         label=f"Mean ROC (AUC={mean_auc:.3f})")

plt.plot([0,1], [0,1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("5-Fold CV ROC Curves")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


# In[ ]:




