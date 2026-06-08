import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Define workspace path
ROOT_DIR = Path("d:/Github projects/Pre-Match-Cricket-Pitch-Condition-Classification")
sys.path.append(str(ROOT_DIR))

from src.config import MODEL_PATH, PREPROCESSOR_PATH, DATASET_PATH, TARGET_COLUMN
from src.preprocess import preprocess_dataset, transform_splits

data = preprocess_dataset()
model = joblib.load(MODEL_PATH)

x_train_trans, x_val_trans, x_test_trans = transform_splits(data)

# Compute train performance
y_train_pred = model.predict(x_train_trans)
train_acc = accuracy_score(data.y_train, y_train_pred)
train_f1 = f1_score(data.y_train, y_train_pred, average="macro")

# Compute test performance
y_test_pred = model.predict(x_test_trans)
test_acc = accuracy_score(data.y_test, y_test_pred)
test_f1 = f1_score(data.y_test, y_test_pred, average="macro")

# Compute 5-fold cross-validation score on train set
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, x_train_trans, data.y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
cv_mean = np.mean(cv_scores)

# Generalization Gap
gen_gap_acc = train_acc - test_acc
gen_gap_f1 = train_f1 - test_f1

# Feature Importances
feature_names = [name.replace("numeric__", "").replace("categorical__", "").replace("encoder__", "") for name in data.feature_names]
importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
top_5_features = importances.head(5)

print("--- METRICS ---")
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Train F1 (Macro): {train_f1:.4f}")
print(f"Test F1 (Macro): {test_f1:.4f}")
print(f"Cross-Val Score (Macro F1): {cv_mean:.4f}")
print(f"Gen Gap Accuracy: {gen_gap_acc:.4f}")
print(f"Gen Gap F1: {gen_gap_f1:.4f}")
print("\n--- TOP 5 FEATURE IMPORTANCES ---")
for feat, val in top_5_features.items():
    print(f"{feat}: {val:.4f}")
