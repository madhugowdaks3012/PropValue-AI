import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Generate synthetic dataset
np.random.seed(42)

cities = ["Bengaluru", "Mumbai", "Chennai", "Delhi", "Hyderabad"]

data = []

for _ in range(500):
    city = np.random.choice(cities)
    sqft = np.random.randint(500, 3000)
    bedrooms = np.random.randint(1, 5)
    age = np.random.randint(0, 20)

    base_price = {
        "Bengaluru": 6000,
        "Mumbai": 12000,
        "Chennai": 5000,
        "Delhi": 8000,
        "Hyderabad": 5500
    }[city]

    price = sqft * base_price + bedrooms * 50000 - age * 20000
    price += np.random.randint(-500000, 500000)

    data.append([city, sqft, bedrooms, age, price])

df = pd.DataFrame(data, columns=["City", "Sqft", "Bedrooms", "Age", "Price"])

X = df.drop("Price", axis=1)
y = df["Price"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(), ["City"])
    ],
    remainder="passthrough"
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

model.fit(X, y)

joblib.dump(model, "model.pkl")

print("Model trained and saved as model.pkl")
