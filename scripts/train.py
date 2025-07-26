import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import Counter
from imblearn.over_sampling import SMOTE
import joblib
import os

# --- 1. Load your labeled data ---
data = pd.read_csv('./final_preprocessed.csv')

# --- 2. Feature engineering ---
# Create 'is_centered' feature
# Estimate global page width as the max x1 in the dataset
page_width = data['x1'].max()
data['page_width'] = page_width


# Compute midpoints
line_midpoint = (data['x0'] + data['x1']) / 2
page_midpoint = page_width / 2

# Feature: is_centered
data['is_centered'] = (abs(line_midpoint - page_midpoint) < 20).astype(int)


line_midpoint = (data['x0'] + data['x1']) / 2
page_midpoint = data['page_width'] / 2
data['is_centered'] = (abs(line_midpoint - page_midpoint) < 20).astype(int)

# --- 3. Define features and target ---
features = [
    'size', 'is_bold', 'is_italic', 'font_encoded', 'color_encoded',
    'x0', 'y0', 'x1', 'y1'
]
target = 'label_encoded'

X = data[features]
y = data[target]

print("Original class distribution:", Counter(y))

# --- 4. Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 5. Handle imbalance with SMOTE ---
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Resampled class distribution:", Counter(y_train_res))

# --- 6. Train LightGBM model ---
lgbm = lgb.LGBMClassifier(
    objective='multiclass',
    class_weight='balanced',
    random_state=42
)

lgbm.fit(X_train_res, y_train_res)

# --- 7. Evaluate ---
y_pred = lgbm.predict(X_test)

print("\nClassification report:")
print(classification_report(y_test, y_pred))

print(f"Model accuracy on test set: {lgbm.score(X_test, y_test):.2f}")

# --- 8. Save model and label mapping ---
unique_labels = sorted(y.unique())
# Replace with real names if you have them (e.g., {0:'BODY', 1:'TITLE', etc.})
label_mapping = {label: str(label) for label in unique_labels}

os.makedirs('model', exist_ok=True)
joblib.dump(lgbm, 'model/heading_model.pkl')
joblib.dump(label_mapping, 'model/label_mapping.pkl')

print("✅ Model and label mapping saved to 'model/' directory.")
