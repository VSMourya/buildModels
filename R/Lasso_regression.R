#' Fit Lasso Linear Regression and Evaluate Accuracy
#'
#' This function fits a Lasso Linear Regression model using the glmnet package and evaluates
#' its R-squared performance on both training and testing datasets. It supports feature selection
#' by choosing top k predictors based on importance.
#'
#' @param data A data frame containing the predictors and response variable.
#' @param predictors A character vector of predictor variable names.
#' @param response The name of the response variable.
#' @param lambda The regularization parameter for Lasso.
#' @param top_k The number of top k predictors to select based on importance; if NULL, uses all predictors.
#' @return A list containing the model, training R-squared, testing R-squared, and used predictors.
#' @examples
#' \dontrun{
#'   data <- data.frame(matrix(rnorm(100 * 10), ncol = 10))
#'   colnames(data) <- paste0("X", 1:10)
#'   data$Y <- rnorm(100)
#'   result <- fit_linear_lasso_regression(data, predictors = paste0("X", 1:10), response = "Y")
#' }
#' @export
fit_linear_lasso_regression <- function(data, predictors, response, lambda = 1, top_k = NULL) {
  if (!requireNamespace("glmnet", quietly = TRUE)) {
    install.packages("glmnet")
    library(glmnet)
  }

  if (!is.data.frame(data)) {
    stop("Data must be a dataframe.")
  }

  data <- na.omit(data)
  if (nrow(data) == 0) {
    stop("No data available after removing NA values.")
  }

  if (!(response %in% names(data))) {
    stop("Response variable not found in the dataframe.")
  }

  if (!all(predictors %in% names(data))) {
    stop("Some predictors not found in the dataframe.")
  }

  if (!is.null(top_k) && !is.na(top_k)) {
    predictors <- select_top_features(data = data, predictors = predictors, response = response, k = top_k)
    cat("Selected top K Features:", predictors, "\n")
  }

  split_result <- train_test_split(data)
  data_train <- split_result$train
  data_test <- split_result$test

  cat("Dimensions of Training Data - Rows:", nrow(data_train), "Columns:", ncol(data_train), "\n")
  cat("Dimensions of Testing Data - Rows:", nrow(data_test), "Columns:", ncol(data_test), "\n")

  # Prepare data for glmnet
  y_train <- data_train[[response]]
  x_train <- as.matrix(data_train[predictors])
  y_test <- data_test[[response]]
  x_test <- as.matrix(data_test[predictors])

  # Fit Lasso Regression model
  lasso_model <- glmnet(x_train, y_train, alpha = 1, lambda = lambda)

  saveRDS(lasso_model, "lasso_regression.rds")

  # Predict and calculate R-squared
  predictions_train <- predict(lasso_model, s = lambda, newx = x_train)
  r_squared_train <- 1 - sum((y_train - predictions_train)^2) / sum((y_train - mean(y_train))^2)

  predictions_test <- predict(lasso_model, s = lambda, newx = x_test)
  r_squared_test <- 1 - sum((y_test - predictions_test)^2) / sum((y_test - mean(y_test))^2)

  cat("R-squared on train data:", r_squared_train, "\n")
  cat("R-squared on test data:", r_squared_test, "\n")
  print("Lasso model is saved in present directory with the name lasso_regression")

  return(list(model = lasso_model, r_squared_train = r_squared_train, r_squared_test = r_squared_test, predictors = predictors))
}
