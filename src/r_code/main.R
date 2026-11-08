library(dplyr)
library(caret)

cat("Starting Titanic Survival Prediction in R...\n")

# Load training data
train <- read.csv("src/data/train.csv",
                  stringsAsFactors = FALSE,
                  na.strings = c("", "NA"))

cat("\n--- Training Dataset ---\n")
cat("Dataset shape:", nrow(train), "rows,", ncol(train), "columns\n")
print(head(train))

cat("\nMissing Values:\n")
print(colSums(is.na(train)))

cat("\nSurvival Distribution:\n")
print(table(train$Survived))


# Feature Engineering
create_features <- function(df) {

  df <- df %>%
    mutate(
      FamilySize = SibSp + Parch + 1,
      IsAlone = as.integer(FamilySize == 1),
      CabinKnown = as.integer(!is.na(Cabin) & Cabin != "")
    )

  # Extract passenger title
  df$Title <- sub(".*?,\\s*([^.]*)\\..*", "\\1", df$Name)

  df$Title[df$Title %in% c("Mlle", "Ms")] <- "Miss"
  df$Title[df$Title == "Mme"] <- "Mrs"

  common_titles <- c("Mr", "Mrs", "Miss", "Master")
  df$Title[!df$Title %in% common_titles] <- "Rare"

  return(df)
}

train <- create_features(train)

features <- c(
  "Pclass", "Sex", "Age", "Fare", "Embarked",
  "FamilySize", "IsAlone", "Title", "CabinKnown"
)

cat("\n--- Feature Engineering ---\n")
cat("Selected features:\n")
print(features)

cat("\nNew Features:\n")
print(head(train[, c("FamilySize", "IsAlone", "Title", "CabinKnown")]))

cat("\nTitle Distribution:\n")
print(table(train$Title))

cat("\nMissing Values in Selected Features:\n")
print(colSums(is.na(train[, features])))


# Train / Validation Split
set.seed(42)

train_index <- createDataPartition(
  train$Survived,
  p = 0.8,
  list = FALSE
)

train_data <- train[train_index, ]
val_data <- train[-train_index, ]

cat("\n--- Train/Validation Split ---\n")
cat("Training samples:", nrow(train_data), "\n")
cat("Validation samples:", nrow(val_data), "\n")


# Handle Missing Values
# Use training data statistics to avoid data leakage
age_median <- median(train_data$Age, na.rm = TRUE)
fare_median <- median(train_data$Fare, na.rm = TRUE)

embarked_mode <- names(
  sort(table(train_data$Embarked[!is.na(train_data$Embarked) &
                                train_data$Embarked != ""]), decreasing = TRUE)
)[1]

preprocess_data <- function(df) {

  df$Age[is.na(df$Age)] <- age_median
  df$Fare[is.na(df$Fare)] <- fare_median

  df$Embarked[is.na(df$Embarked) | df$Embarked == ""] <- embarked_mode

  # Convert categorical features into factors
  categorical <- c(
    "Pclass", "Sex", "Embarked", "Title",
    "IsAlone", "CabinKnown"
  )

  for (col in categorical) {
    df[[col]] <- as.factor(df[[col]])
  }

  return(df)
}

train_data <- preprocess_data(train_data)
val_data <- preprocess_data(val_data)

# Keep factor levels consistent across datasets
categorical <- c(
  "Pclass", "Sex", "Embarked", "Title",
  "IsAlone", "CabinKnown"
)

for (col in categorical) {
  val_data[[col]] <- factor(
    val_data[[col]],
    levels = levels(train_data[[col]])
  )
}

cat("\n--- Missing Values After Preprocessing ---\n")
print(colSums(is.na(train_data[, features])))


# Logistic Regression
formula <- as.formula(
  paste("Survived ~", paste(features, collapse = " + "))
)

model <- glm(
  formula,
  data = train_data,
  family = binomial
)

cat("\n--- Logistic Regression ---\n")
cat("Model trained successfully!\n")


# Training Accuracy
train_prob <- predict(
  model,
  newdata = train_data,
  type = "response"
)

train_predictions <- ifelse(train_prob >= 0.5, 1, 0)

train_accuracy <- mean(
  train_predictions == train_data$Survived
)

cat("\n--- Training Accuracy ---\n")
cat(sprintf("Training Accuracy: %.4f\n", train_accuracy))


# Validation Accuracy
val_prob <- predict(
  model,
  newdata = val_data,
  type = "response"
)

val_predictions <- ifelse(val_prob >= 0.5, 1, 0)

val_accuracy <- mean(
  val_predictions == val_data$Survived
)

cat("\n--- Validation Accuracy ---\n")
cat(sprintf("Validation Accuracy: %.4f\n", val_accuracy))


# Load Test Dataset
test <- read.csv("src/data/test.csv",
                 stringsAsFactors = FALSE,
                 na.strings = c("", "NA"))

cat("\n--- Test Dataset ---\n")
cat("Test dataset shape:", nrow(test), "rows,", ncol(test), "columns\n")

test <- create_features(test)
test <- preprocess_data(test)

for (col in categorical) {
  test[[col]] <- factor(
    test[[col]],
    levels = levels(train_data[[col]])
  )
}


# Predict Test Dataset
test_prob <- predict(
  model,
  newdata = test,
  type = "response"
)

test_predictions <- ifelse(test_prob >= 0.5, 1, 0)

cat("\n--- Test Predictions ---\n")
cat("Number of predictions:", length(test_predictions), "\n")
cat("First 10 predictions:\n")
print(head(test_predictions, 10))

results <- data.frame(
  PassengerId = test$PassengerId,
  Survived = test_predictions
)

cat("\nFirst 5 prediction results:\n")
print(head(results, 5))

cat("\nTitanic Survival Prediction in R Completed!\n")