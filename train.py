import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression

# -----------------------
# Load Dataset
# -----------------------

df = pd.read_csv("data/Titanic.csv")

# Keep only required columns
df = df[["Pclass", "Sex", "Age", "Fare", "Survived"]]

X = df.drop("Survived", axis=1)
y = df["Survived"]

# -----------------------
# Features
# -----------------------

numeric_features = ["Age", "Fare"]
categorical_features = ["Sex", "Pclass"]

# -----------------------
# Numeric Pipeline
# -----------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# -----------------------
# Categorical Pipeline
# -----------------------

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder())
])

# -----------------------
# Combine
# -----------------------

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# -----------------------
# Final Pipeline
# -----------------------

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression())
])

pipeline.fit(X, y)

joblib.dump(pipeline, "model/pipeline.pkl")

print("Pipeline saved successfully.")