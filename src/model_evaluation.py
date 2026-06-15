import numpy as np
import pandas as pd
import os
import json

import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score

xgb_model = pickle.load(open("models/xgb_model.pkl", "rb"))

test_data = pd.read_csv(os.path.join("data", "feature_engineered", "test_feature_engineered.csv"))
X_test = test_data.drop(columns=['label'])
y_test = test_data['label'].map({'sadness': 0, 'happiness': 1})

# Make predictions
y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "auc": auc
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)