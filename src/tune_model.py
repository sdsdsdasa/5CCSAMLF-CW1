import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Set seed
np.random.seed(123)

# Import training data
trn = pd.read_csv('data/CW1_train.csv')
X_tst = pd.read_csv('data/CW1_test.csv')  # no outcomes

# Identify categorical columns
categorical_cols = ['cut', 'color', 'clarity']

# One-hot encode categorical variables (baseline)
trn = pd.get_dummies(trn, columns=categorical_cols, drop_first=True)
X_tst = pd.get_dummies(X_tst, columns=categorical_cols, drop_first=True)

# Split X/y
X = trn.drop(columns=['outcome'])
y = trn['outcome']

# IMPORTANT: align columns between train and test (prevents mismatch)
X, X_tst = X.align(X_tst, join="left", axis=1, fill_value=0)

# Train/validation split for model selection (out-of-sample estimate)
X_trn, X_val, y_trn, y_val = train_test_split(X, y, test_size=0.2, random_state=123)

# Define models to compare (simple set)
models = {
    "RandomForest0": RandomForestRegressor( 
        random_state=123, 
        n_jobs=-1, 
        n_estimators=300, #[300, 600]
        max_depth=None, #[None, 10, 20]
        min_samples_split=2, #[2, 5]
        min_samples_leaf=1 #[1, 2]
    ),
    "RandomForest1": RandomForestRegressor( 
        random_state=123, 
        n_jobs=-1, 
        n_estimators=600, #[300, 600]
        max_depth=None, #[None, 10, 20]
        min_samples_split=2, #[2, 5]
        min_samples_leaf=1 #[1, 2]
    ),
    "RandomForest2": RandomForestRegressor( 
        random_state=123, 
        n_jobs=-1, 
        n_estimators=300, #[300, 600]
        max_depth=10, #[None, 10, 20]
        min_samples_split=2, #[2, 5]
        min_samples_leaf=1 #[1, 2]
    ),
    "RandomForest3": RandomForestRegressor( 
        random_state=123, 
        n_jobs=-1, 
        n_estimators=300, #[300, 600]
        max_depth=20, #[None, 10, 20]
        min_samples_split=2, #[2, 5]
        min_samples_leaf=1 #[1, 2]
    ),
    "RandomForest4": RandomForestRegressor( 
        random_state=123, 
        n_jobs=-1, 
        n_estimators=300, #[300, 600]
        max_depth=None, #[None, 10, 20]
        min_samples_split=5, #[2, 5]
        min_samples_leaf=1 #[1, 2]
    ),
    "RandomForest5": RandomForestRegressor( 
        random_state=123, 
        n_jobs=-1, 
        n_estimators=300, #[300, 600]
        max_depth=None, #[None, 10, 20]
        min_samples_split=2, #[2, 5]
        min_samples_leaf=2 #[1, 2]
    ),
    "GradientBoosting0": GradientBoostingRegressor(
        random_state=123, 
        n_estimators=200, #[200, 400]
        learning_rate=0.05, #[0.05, 0.1]
        max_depth=2, #[2, 3, 4]
        subsample=0.8 #[0.8, 1.0]
    ),
    "GradientBoosting1": GradientBoostingRegressor(
        random_state=123, 
        n_estimators=400, #[200, 400]
        learning_rate=0.05, #[0.05, 0.1]
        max_depth=2, #[2, 3, 4]
        subsample=0.8 #[0.8, 1.0]
    ),
    "GradientBoosting2": GradientBoostingRegressor(
        random_state=123, 
        n_estimators=200, #[200, 400]
        learning_rate=0.1, #[0.05, 0.1]
        max_depth=2, #[2, 3, 4]
        subsample=0.8 #[0.8, 1.0]
    ),
    "GradientBoosting3": GradientBoostingRegressor(
        random_state=123, 
        n_estimators=200, #[200, 400]
        learning_rate=0.05, #[0.05, 0.1]
        max_depth=3, #[2, 3, 4]
        subsample=0.8 #[0.8, 1.0]
    ),
    "GradientBoosting4": GradientBoostingRegressor(
        random_state=123, 
        n_estimators=200, #[200, 400]
        learning_rate=0.05, #[0.05, 0.1]
        max_depth=4, #[2, 3, 4]
        subsample=0.8 #[0.8, 1.0]
    ),
    "GradientBoosting5": GradientBoostingRegressor(
        random_state=123, 
        n_estimators=200, #[200, 400]
        learning_rate=0.05, #[0.05, 0.1]
        max_depth=2, #[2, 3, 4]
        subsample=1.0 #[0.8, 1.0]
    ),
    
    
}

# Fit + evaluate each model on validation set
results = []
for name, model in models.items():
    model.fit(X_trn, y_trn)
    yhat_val = model.predict(X_val)
    r2 = r2_score(y_val, yhat_val)
    results.append((name, r2))
    print(f"{name:25s}  R^2 (val) = {r2:.4f}")

# Show ranking
results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
print("\n=== Model ranking (best -> worst) ===")
for name, r2 in results_sorted:
    print(f"{name:25s}  {r2:.4f}")


#--------------------------------------#

# Train best model on FULL training data and produce test predictions
best_name = results_sorted[0][0]
best_model = models[best_name]
best_model.fit(X, y)
yhat_test = best_model.predict(X_tst)

# Save submission in required format
out = pd.DataFrame({'yhat': yhat_test})
out.to_csv('outputs/CW1_submission_K23158987.csv', index=False)

print(f"\nSaved submission using best model: {best_name}")
