import pandas as pd
import os
import pickle
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score

# fetch the data from data/feature_engineered folder
train_data = pd.read_csv(os.path.join("data", "feature_engineered", "train_feature_engineered.csv"))


X_train = train_data.drop(columns=['label'])
y_train = train_data['label'].map({'sadness': 0, 'happiness': 1})

# Train the XGBoost model
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

#save the model in models folder
os.makedirs("models", exist_ok=True)
pickle.dump(xgb_model, open(os.path.join("models", "xgb_model.pkl"), "wb"))