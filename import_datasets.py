from sklearn.datasets import fetch_openml
import pandas as pd

X, y = fetch_openml(
    "ringnorm",
    version=1,
    return_X_y=True,
    as_frame=True
)

df = X.copy()

df.to_csv("data.csv", index=True)
y.to_csv("label.csv", index=True)