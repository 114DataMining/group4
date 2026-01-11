#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

# ============================================================
# 0) 讀取資料 + 基本清理
# ============================================================
DATA_PATH = "vgsales_clean.csv"

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.lower()

# 欄位內容清理
df["platform"]  = df["platform"].astype(str).str.strip().str.upper()
df["genre"]     = df["genre"].astype(str).str.strip()
df["publisher"] = df["publisher"].astype(str).str.strip()

# 丟掉必要欄位缺值
need_cols = ["platform", "genre", "publisher", "year", "global_sales"]
df = df.dropna(subset=need_cols).copy()

df["year"] = df["year"].astype(int)
df["global_sales"] = pd.to_numeric(df["global_sales"], errors="coerce")
df = df.dropna(subset=["global_sales"]).copy()

print("原始資料 shape:", df.shape)

# ============================================================
# 1) 平台分群（SONY / NINTENDO / XBOX / PC / OTHER）
# ============================================================
SONY = {"PS", "PS2", "PS3", "PS4", "PSP", "PSV"}
NINTENDO = {"3DS", "DS", "GB", "GBA", "GC", "N64", "NES", "SNES", "WII", "WIIU"}
XBOX = {"XB", "X360", "XONE"}
PCSET = {"PC"}

def platform_group(p):
    if p in SONY:
        return "SONY"
    if p in NINTENDO:
        return "NINTENDO"
    if p in XBOX:
        return "XBOX"
    if p in PCSET:
        return "PC"
    return "OTHER"

df["platform_group"] = df["platform"].apply(platform_group)

print("\n平台分群後分布：")
print(df["platform_group"].value_counts())

# ============================================================
# 2) 目標變數 Hit（以 Global_Sales 中位數切 0/1）
# ============================================================
median_sales = df["global_sales"].median()
df["hit"] = (df["global_sales"] >= median_sales).astype(int)

print("\nHit 分布：")
print(df["hit"].value_counts())
print("Hit 比例：")
print(df["hit"].value_counts(normalize=True))

# ============================================================
# 3) Top10 Publisher（二元特徵）
# ============================================================
top_publishers = (
    df.groupby("publisher")["hit"].sum()
      .sort_values(ascending=False)
      .head(10)
      .index
)

df["top10_publisher"] = df["publisher"].isin(top_publishers).astype(int)

print("\nTop10 Publisher 命中比例：", df["top10_publisher"].mean())

# ============================================================
# 4) Genre：Top5 + Others
# ============================================================
top5_genres = df["genre"].value_counts().head(5).index.tolist()
df["genre_group"] = np.where(df["genre"].isin(top5_genres), df["genre"], "Others")

print("\nGenre 分布（Top5 + Others）：")
print(df["genre_group"].value_counts())

# ============================================================
# 5) 最終資料（未編碼）
# ============================================================
df_final = df[[
    "platform_group",
    "genre_group",
    "top10_publisher",
    "hit"
]].copy()

print("\n最終資料 shape:", df_final.shape)
print("最終欄位：", df_final.columns.tolist())

# ============================================================
# 6) One-Hot Encoding（老師最在意）
# ============================================================
X = df_final[["platform_group", "genre_group", "top10_publisher"]]
y = df_final["hit"]

X_encoded = pd.get_dummies(
    X,
    columns=["platform_group", "genre_group"],
    drop_first=False
)

df_onehot = X_encoded.copy()
df_onehot["hit"] = y.values

print("\nOne-Hot 後特徵數：", X_encoded.shape[1])
print("One-Hot 欄位清單：")
print(X_encoded.columns.tolist())

# ============================================================
# 7) 輸出檔案
# ============================================================
df_final.to_csv("vgsales_focus_final.csv", index=False)
df_onehot.to_csv("vgsales_focus_onehot.csv", index=False)

print("\n已輸出：")
print("vgsales_focus_final.csv  （未編碼、可解釋）")
print("vgsales_focus_onehot.csv （One-Hot 後、模型用）")


# In[ ]:




