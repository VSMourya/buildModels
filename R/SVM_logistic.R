#' Fit an SVM Logistic Regression Model and Evaluate Accuracy
#'
#' This function fits an SVM Logistic Regression model using the `e1071` package with a radial kernel.
#' It handles package installation/loading, converts the response variable into a binary factor, checks for
#' binary nature, optionally selects top features, fits the model, and calculates accuracy for both
#' training and testing datasets. The model is saved to the current directory.
#'
#' @param data A dataframe containing the dataset.
#' @param predictors A character vector of predictor variable names.
#' @param response A string indicating the response variable name.
#' @param top_k Optional; an integer for selecting top K features based on importance.
#'
#' @return A list containing the fitted SVM model, training accuracy, testing accuracy, and the predictors used.
#' @import e1071
#' @examples
#' data(iris)
#' iris$Species <- ifelse(iris$Species == "setosa", 1, 0)
#' result <- fit_logistic_svm(data = iris, predictors = c("Sepal.Length", "Sepal.Width"), response = "Species")
#' print(result)
#' @export

fit_logistic_svm <- function(data, predictors, response, top_k = NULL) {
  # Ensure the 'e1071' package is installed and loaded
  if (!require(e1071, quietly = TRUE)) {
    install.packages("e1071")
    library(e1071)
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

  # Split data into training and testing sets
  split_result <- train_test_split(data)
  data_train <- split_result$train
  data_test <- split_result$test

  # Fit SVM model with a radial kernel
  formula_svm <- as.formula(paste(response, "~", paste(predictors, collapse = "+")))
  svm_model <- svm(formula_svm, data = data_train, kernel = "radial")
  saveRDS(svm_model, "logistic_svm.rds")

  # Predict and calculate accuracy for training data
  train_predictions <- predict(svm_model, data_train)
  train_accuracy <- mean(train_predictions == data_train[[response]])

  # Predict and calculate accuracy for testing data
  test_predictions <- predict(svm_model, data_test)
  test_accuracy <- mean(test_predictions == data_test[[response]])

  cat(sprintf("Training Accuracy: %.4f, Testing Accuracy: %.4f\n", train_accuracy, test_accuracy))
  cat("SVM Logistic Regression model saved as 'logistic_svm.rds'\n")

  return(list(model = svm_model, accuracy_train = train_accuracy, accuracy_test = test_accuracy, predictors = predictors))
}
