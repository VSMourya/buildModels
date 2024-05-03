#' Fit a Ridge Logistic Regression Model and Evaluate Accuracy
#'
#' This function fits a Ridge Logistic Regression model using the `glmnet` package. It handles
#' package installation/loading, data preparation, feature selection, model fitting, and accuracy
#' calculation. It removes NA values, checks for proper data formatting, splits the data into
#' training and testing sets, fits the model, saves it, and evaluates its accuracy on both sets.
#'
#' @param data A dataframe containing the dataset.
#' @param predictors A character vector of predictor variable names.
#' @param response A string indicating the response variable name.
#' @param lambda Regularization parameter; default is 1.
#' @param top_k Optional; an integer specifying the number of top features to select based on importance.
#'
#' @return A list containing the fitted model, training accuracy, testing accuracy, and the predictors used.
#' @import glmnet
#' @examples
#' data(iris)
#' iris$Species <- ifelse(iris$Species == "setosa", 1, 0)
#' result <- fit_logistic_ridge(data = iris, predictors = c("Sepal.Length", "Sepal.Width"), response = "Species")
#' print(result)
#' @export

fit_logistic_ridge <- function(data, predictors, response, lambda = 1, top_k = NULL) {
  # Ensure the 'glmnet' package is installed and loaded
  if (!require(glmnet, quietly = TRUE)) {
    install.packages("glmnet")
    library(glmnet)
  }

  # Handle missing data and ensure correct data structure
  data <- na.omit(data)
  if (!is.data.frame(data)) {
    stop("Data must be a dataframe.")
  }
  if (!(response %in% names(data))) {
    stop("Predictor or response not found in the dataframe.")
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

  # Prepare data matrices for glmnet
  x_train <- as.matrix(data_train[predictors])
  y_train <- data_train[[response]]
  x_test <- as.matrix(data_test[predictors])
  y_test <- data_test[[response]]

  # Fit Ridge Logistic Regression model
  ridge_model <- glmnet(x_train, y_train, alpha = 0, family = "binomial", lambda = lambda)
  saveRDS(ridge_model, "ridge_logistic.rds")

  # Predict and calculate accuracy
  train_predictions <- predict(ridge_model, newx = x_train, s = lambda, type = "response")
  test_predictions <- predict(ridge_model, newx = x_test, s = lambda, type = "response")
  train_accuracy <- mean((train_predictions > 0.5) == y_train)
  test_accuracy <- mean((test_predictions > 0.5) == y_test)

  cat(sprintf("Training Accuracy: %.4f, Testing Accuracy: %.4f\n", train_accuracy, test_accuracy))
  cat("Ridge Logistic Regression model saved as 'ridge_logistic.rds'\n")

  return(list(model = ridge_model, accuracy_train = train_accuracy, accuracy_test = test_accuracy, predictors = predictors))
}
