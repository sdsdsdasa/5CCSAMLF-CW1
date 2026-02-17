import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

np.random.seed(123)

# ------------------------- Load & preprocess -------------------------
trn = pd.read_csv('data/CW1_train.csv')
X_tst = pd.read_csv('data/CW1_test.csv')

categorical_cols = ['cut', 'color', 'clarity']

trn = pd.get_dummies(trn, columns=categorical_cols, drop_first=True)
X_tst = pd.get_dummies(X_tst, columns=categorical_cols, drop_first=True)

X = trn.drop(columns=['outcome'])
y = trn['outcome']

# IMPORTANT: align train/test columns
X, X_tst = X.align(X_tst, join="left", axis=1, fill_value=0)

# Hold-out validation split (out-of-sample estimate)
X_trn, X_val, y_trn, y_val = train_test_split(X, y, test_size=0.2, random_state=123)

# ------------------------- Define models + tuning grids -------------------------
candidates = [
    (
        "RandomForest",
        RandomForestRegressor(random_state=123, n_jobs=-1),
        {
            "n_estimators": [300, 600],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        }
    ),
    (
        "GradientBoosting",
        GradientBoostingRegressor(random_state=123),
        {
            "n_estimators": [200, 400],
            "learning_rate": [0.05, 0.1],
            "max_depth": [2, 3, 4],
            "subsample": [0.8, 1.0]
        }
    )
]

# ------------------------- Tune each model using CV on the training split only -------------------------
best_overall = None
best_score = -1e9
best_name = None

results = []

for name, model, grid in candidates:
    gs = GridSearchCV(
        estimator=model,
        param_grid=grid,
        scoring="r2",
        cv=5,              # cross-validation INSIDE training split
        n_jobs=-1
    )
    gs.fit(X_trn, y_trn)

    # Evaluate the tuned model on the hold-out validation set
    best_model = gs.best_estimator_
    yhat_val = best_model.predict(X_val)
    r2_val = r2_score(y_val, yhat_val)

    results.append((name, gs.best_params_, gs.best_score_, r2_val))
    print(f"\n{name}")
    print("  Best CV R^2 (on train split):", round(gs.best_score_, 4))
    print("  Best params:", gs.best_params_)
    print("  Hold-out VAL R^2:", round(r2_val, 4))

    if r2_val > best_score:
        best_score = r2_val
        best_overall = best_model
        best_name = name

# Rank by validation score
results_sorted = sorted(results, key=lambda x: x[3], reverse=True)
print("\n=== Ranked by hold-out validation R^2 ===")
for name, params, cv_r2, val_r2 in results_sorted:
    print(f"{name:18s}  VAL R^2={val_r2:.4f}  CV(best)={cv_r2:.4f}  params={params}")

print(f"\nSelected final model: {best_name} (VAL R^2={best_score:.4f})")

# ------------------------- Retrain selected model on FULL training data, predict test, save submission -------------------------
best_overall.fit(X, y)
yhat_test = best_overall.predict(X_tst)

out = pd.DataFrame({"yhat": yhat_test})
out.to_csv("outputs/CW1_submission_K23158987.csv", index=False)
print("Saved:", "outputs/CW1_submission_K23158987.csv")
