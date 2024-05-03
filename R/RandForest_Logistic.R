#' Fit a Random Forest Model for Binary Logistic Regression and Evaluate Accuracy
#'
#' This function fits a Random Forest logistic regression model using the `randomForest` package.
#' It ensures the package is installed and loaded, converts the response variable to a binary factor,
#' checks for binary nature, optionally selects top features based on importance, fits the model,
#' and computes accuracy for both training and testing datasets. The model is saved in the working directory.
#'
#' @param data A data frame containing the training dataset.
#' @param predictors A character vector of predictor variable names.
#' @param response A string indicating the response variable name.
#' @param top_k Optional; an integer for selecting top K features based on importance.
#'
#' @return A list containing the fitted model, training accuracy, testing accuracy, and the predictors used.
#' @import randomForest
#' @examples
#' data(iris)
#' iris$Species <- ifelse(iris$Species == "setosa", 1, 0)
#' result <- fit_random_forest_binary(data = iris, predictors = c("Sepal.Length", "Sepal.Width"), response = "Species")
#' print(result)
#' @export

fit_random_forest_binary <- function(data, predictors, response, top_k = NULL) {
  # Ensure the 'randomForest' package is installed and loaded
  if (!require(randomForest, quietly = TRUE)) {
    install.packages("randomForest")
    library(randomForest)
  }

  # Convert response to factor and check if binary
  data[[response]] <- factor(data[[response]], levels = c(0, 1))
  if (!is.factor(data[[response]]) || length(levels(data[[response]])) != 2) {
    stop("Response variable must be binary (factor with 2 levels).")
  }

  # Optional: Select top K features based on importance
  if (!is.null(top_k)) {
    predictors <- select_top_features(data = data, predictors = predictors, response = response, k = top_k)
    cat("Selected top K Features:\n", paste(predictors, collapse=", "), "\n")
  }

  # Split the dataset into training and testing subsets
  split_result <- train_test_split(data)
  data_train <- split_result$train
  data_test <- split_result$test

  # Fit the Random Forest model
  formula_rf <- as.formula(paste(response, "~", paste(predictors, collapse = "+")))
  rf_model <- randomForest(formula_rf, data = data_train)

  # Save the model
  saveRDS(rf_model, "random_forest_binary.rds")

  # Predict and calculate accuracy
  train_accuracy <- mean(predict(rf_model, data_train) == data_train[[response]])
  test_accuracy <- mean(predict(rf_model, data_test) == data_test[[response]])

  cat(sprintf("Training Accuracy: %.4f, Testing Accuracy: %.4f\n", train_accuracy, test_accuracy))
  cat("Random Forest model saved as 'random_forest_binary.rds'\n")

  return(list(model = rf_model, accuracy_train = train_accuracy, accuracy_test = test_accuracy, predictors = predictors))
}
