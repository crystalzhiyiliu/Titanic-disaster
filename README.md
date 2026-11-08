# Titanic Disaster

This project uses Logistic Regression in Python and R to predict Titanic passenger survival. Both models run in Docker containers for reproducibility.

## Repository Structure

```text
Titanic-disaster/
├── .github/
│   └── CODEOWNERS
├── .gitignore
├── README.md
└── src/
    ├── data/
    │   └── .gitkeep
    ├── python_code/
    │   ├── Dockerfile
    │   ├── main.py
    │   └── requirements.txt
    └── r_code/
        ├── Dockerfile
        ├── install_packages.R
        └── main.R
```

## Prerequisites

- Git
- Docker Desktop (installed and running)
- Kaggle account

Python and R do not need to be installed locally.

## Clone the Repository

```bash
git clone https://github.com/crystalzhiyiliu/Titanic-disaster.git
cd Titanic-disaster
```

## Download the Data

1. Download `train.csv` and `test.csv` from the [Kaggle Titanic Dataset](https://www.kaggle.com/competitions/titanic/data).
2. Place both files in `src/data/`.

If the directory does not exist:

```bash
mkdir -p src/data
```

Expected structure:

```text
src/data/
├── .gitkeep
├── train.csv
└── test.csv
```

The CSV files are excluded from GitHub.

## Run the Python Container

### Build

Run from the repository root:

```bash
docker build -f src/python_code/Dockerfile -t titanic-python .
```

### Run

```bash
docker run --rm -v "$(pwd)/src/data:/app/src/data:ro" titanic-python
```

### What the Code Does

- Loads and explores the Titanic dataset
- Handles missing values
- Creates `FamilySize`, `IsAlone`, `Title`, and `CabinKnown`
- Encodes categorical features and scales numerical features
- Splits the data into training and validation sets (80/20)
- Trains Logistic Regression using `scikit-learn`
- Evaluates model accuracy and predicts survival for 418 test passengers

## Run the R Container

### Build

Run from the repository root:

```bash
docker build -f src/r_code/Dockerfile -t titanic-r .
```

### Run

```bash
docker run --rm -v "$(pwd)/src/data:/app/src/data:ro" titanic-r
```

### What the Code Does

- Loads and explores the Titanic dataset
- Handles missing values
- Creates `FamilySize`, `IsAlone`, `Title`, and `CabinKnown`
- Prepares numerical and categorical features
- Splits the data into training and validation sets (80/20)
- Trains Logistic Regression using R's `glm()`
- Evaluates model accuracy and predicts survival for 418 test passengers

## Model Results

| Metric | Python | R |
|---|---:|---:|
| Training Accuracy | 83.43% | 84.85% |
| Validation Accuracy | 82.68% | 79.21% |
| Test Predictions | 418 | 418 |

Both models use nine features: `Pclass`, `Sex`, `Age`, `Fare`, `Embarked`, `FamilySize`, `IsAlone`, `Title`, and `CabinKnown`.

Results and predictions are printed in the terminal. The models do not automatically save prediction CSV files.

The Kaggle test dataset does not include survival labels, so accuracy is evaluated using a validation split from `train.csv`.