import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


print("Starting Titanic Survival Prediction...")


# Load training dataset
train = pd.read_csv("src/data/train.csv")

print("\n--- Training Dataset ---")
print("Dataset shape:", train.shape)
print(train.head())

print("\nMissing Values:")
print(train.isnull().sum())

print("\nSurvival Distribution:")
print(train["Survived"].value_counts())


# Feature Engineering
def create_features(df):
    df = df.copy()

    # Family-related features
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Extract titles from names
    df["Title"] = df["Name"].str.extract(r",\s*([^.]*)\.")

    # Combine similar titles
    df["Title"] = df["Title"].replace({
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs"
    })

    # Group uncommon titles
    common_titles = ["Mr", "Mrs", "Miss", "Master"]
    df["Title"] = df["Title"].where(
        df["Title"].isin(common_titles), "Rare"
    )

    # Whether cabin information is available
    df["CabinKnown"] = df["Cabin"].notna().astype(int)

    return df


train = create_features(train)

features = [
    "Pclass",
    "Sex",
    "Age",
    "Fare",
    "Embarked",
    "FamilySize",
    "IsAlone",
    "Title",
    "CabinKnown"
]

print("\n--- Feature Engineering ---")
print("Selected features:", features)

print("\nNew Features:")
print(train[["FamilySize", "IsAlone", "Title", "CabinKnown"]].head())

print("\nTitle Distribution:")
print(train["Title"].value_counts())

print("\nMissing Values in Selected Features:")
print(train[features].isnull().sum())


# Train/Test Split
X = train[features]
y = train["Survived"]

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n--- Train/Validation Split ---")
print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))


# Data Preprocessing
numeric_features = [
    "Age",
    "Fare",
    "FamilySize"
]

categorical_features = [
    "Pclass",
    "Sex",
    "Embarked",
    "Title",
    "IsAlone",
    "CabinKnown"
]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])


# Train Logistic Regression
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])

model.fit(X_train, y_train)

print("\n--- Logistic Regression ---")
print("Model trained successfully!")


# Training Accuracy
train_predictions = model.predict(X_train)
train_accuracy = accuracy_score(y_train, train_predictions)

print("\n--- Training Accuracy ---")
print(f"Training Accuracy: {train_accuracy:.4f}")


# Validation Accuracy
val_predictions = model.predict(X_val)
val_accuracy = accuracy_score(y_val, val_predictions)

print("\n--- Validation Accuracy ---")
print(f"Validation Accuracy: {val_accuracy:.4f}")


# Predict on Test Dataset
test = pd.read_csv("src/data/test.csv")

print("\n--- Test Dataset ---")
print("Test dataset shape:", test.shape)

test = create_features(test)

test_predictions = model.predict(test[features])

print("\n--- Test Predictions ---")
print("Number of predictions:", len(test_predictions))
print("First 10 predictions:", test_predictions[:10])


# Create prediction output
results = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": test_predictions
})

print("\nFirst 5 prediction results:")
print(results.head())

print("\nTitanic Survival Prediction Completed!")
