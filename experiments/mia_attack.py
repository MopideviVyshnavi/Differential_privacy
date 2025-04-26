import numpy as np
import joblib
from pathlib import Path
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from privacy_estimates.utils import AttackResults

# Load MNIST or Digits dataset
X, y = load_digits(return_X_y=True)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Load trained model from experiments folder
model_path = Path(__file__).parent / "mnist_model.pkl"
model = joblib.load(model_path)

# Membership Inference Attack Simulation
train_preds = model.predict_proba(X_train).max(axis=1)
test_preds = model.predict_proba(X_test).max(axis=1)

# Scores and Labels
scores = np.concatenate([train_preds, test_preds])
labels = np.concatenate([np.ones(len(train_preds)), np.zeros(len(test_preds))])

# Threshold: Simple Mean of Scores
threshold = np.mean(scores)

# Attack Results
attack_results = AttackResults.from_scores_threshold_and_labels(scores, threshold, labels)

print("Membership Inference Attack Results")
print("===================================")
print(f"True Positives (TP) : {attack_results.TP}")
print(f"True Negatives (TN) : {attack_results.TN}")
print(f"False Positives (FP): {attack_results.FP}")
print(f"False Negatives (FN): {attack_results.FN}")
print(f"Accuracy : {attack_results.accuracy * 100:.2f}%")
