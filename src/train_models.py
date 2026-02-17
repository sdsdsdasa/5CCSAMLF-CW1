import pandas as pd
from sklearn.linear_model import Ridge
from preprocess import split_X_y, fit_transform, transform, PreprocessConfig

train_df = pd.read_csv("data/CW1_train.csv")
test_df = pd.read_csv("data/CW1_test.csv")

X_train, y_train = split_X_y(train_df)

config = PreprocessConfig(
    numeric_transform="none",   # try "yeo_johnson" later if needed
    scale_numeric=True
)

pre, X_train_p = fit_transform(X_train, config)
X_test_p = transform(pre, test_df)

model = Ridge(alpha=1.0)
model.fit(X_train_p, y_train)

yhat = model.predict(X_test_p)

pd.DataFrame({"yhat": yhat}).to_csv("CW1_submission_K23158987.csv", index=False)
