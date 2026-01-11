import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ============================================================
# 1) 全資料集：數字統整
# ============================================================
y_full = df_onehot["hit"].copy()

total_n = len(y_full)
hit1_n = int((y_full == 1).sum())
hit0_n = int((y_full == 0).sum())
hit_rate = y_full.mean()

print("=== Full Dataset Summary ===")
print(f"Total samples: {total_n}")
print(f"HIT=1 count : {hit1_n}")
print(f"HIT=0 count : {hit0_n}")
print(f"HIT rate    : {hit_rate:.4f}")

# ============================================================
# 2) 全資料集：長條圖（HIT=1 vs HIT=0）
# ============================================================
plt.figure()
plt.bar(["HIT=0", "HIT=1"], [hit0_n, hit1_n])
plt.ylabel("Count")
plt.title("Full Dataset: HIT Class Counts")
plt.show()


plt.figure()
plt.bar(["HIT=0", "HIT=1"], [hit0_n/total_n, hit1_n/total_n])
plt.ylim(0, 1)
plt.ylabel("Proportion")
plt.title("Full Dataset: HIT Class Proportions")
plt.show()

# ============================================================
# 3) 切分 Train/Test（分層抽樣）
# ============================================================
X = df_onehot.drop(columns=["hit"]).copy()
y = y_full

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# ============================================================
# 4) Train/Test：用數據表對照（Count + Rate）
# ============================================================
summary_df = pd.DataFrame({
    "Dataset": ["Train", "Test"],
    "Sample Size": [len(y_train), len(y_test)],
    "HIT=1 Count": [int((y_train == 1).sum()), int((y_test == 1).sum())],
    "HIT=0 Count": [int((y_train == 0).sum()), int((y_test == 0).sum())],
    "HIT Rate": [y_train.mean(), y_test.mean()]
})

print("\n=== Train/Test Summary (Stratified Split) ===")
print(summary_df.to_string(index=False))

# 額外：顯示比例差（越接近 0 越好）
diff = abs(y_train.mean() - y_test.mean())
print(f"\nHIT Rate 的 絕對差值(Train vs Test): {diff:.6f}")

# ============================================================
# 5) 長條圖 A：Train vs Test 的 HIT 比例（HIT rate）
# ============================================================
plt.figure()
plt.bar(["Train", "Test"], [y_train.mean(), y_test.mean()])
plt.ylim(0, 1)
plt.ylabel("HIT Rate")
plt.title("HIT Rate Comparison: Train vs Test")
plt.show()

# ============================================================
# 6) 長條圖 B：Train/Test 類別分布（HIT=0/1 比例一起看）
# ============================================================
train_counts = y_train.value_counts(normalize=True).sort_index()
test_counts  = y_test.value_counts(normalize=True).sort_index()

dist_df = pd.DataFrame({
    "Train": train_counts,
    "Test": test_counts
}).fillna(0)

print("\n=== Class Proportion (0/1) ===")
print(dist_df)

# 這張圖會畫出兩組並排長條：0類與1類在 Train/Test 的比例
dist_df.plot(kind="bar")
plt.ylim(0, 1)
plt.ylabel("Proportion")
plt.title("Class Distribution Comparison (Train vs Test)")
plt.show()
