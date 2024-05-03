#' Fit Linear Regression and Evaluate Accuracy
#'
#' This function fits a linear regression model using the base R stats package and evaluates
#' its performance using R-squared on both training and testing datasets. It supports feature selection
#' by selecting top k predictors based on importance.
#'
#' @param data A data frame containing the predictors and response variable.
#' @param predictor A character vector of predictor variable names.
#' @param response The name of the response variable.
#' @param top_k The number of top k predictors to select based on importance; if NULL, uses all predictors.
#' @return A list containing the model, training R-squared, testing R-squared, predictions, and model summary.
#' @examples
#' \dontrun{
#'   data <- data.frame(matrix(rnorm(100 * 10), ncol = 10))
#'   colnames(data) <- paste0("X", 1:10)
#'   data$Y <- rnorm(100)
#'   result <- fit_linear_regression(data, predictors = paste0("X", 1:10), response = "Y")
#' }
#' @export

fit_linear_regression <- function(data, predictor, response, top_k = NULL) {
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

  if (!all(predictor %in% names(data))) {
    stop("Some predictors not found in the dataframe.")
  }

  if (length(unique(data[[response]])) < 2) {
    stop("The response variable does not have enough variability for modeling.")
  }

  if (!is.null(top_k) && !is.na(top_k)) {
    predictor <- select_top_features(data = data, predictors = predictor, response = response, k = top_k)
    cat("Selected top K Features:", predictor, "\n")
  }

  split_result <- train_test_split(data)
  data_train <- split_result$train
  data_test <- split_result$test

  cat("Dimensions of Training Data - Rows:", nrow(data_train), "Columns:", ncol(data_train), "\n")
  cat("Dimensions of Testing Data - Rows:", nrow(data_test), "Columns:", ncol(data_test), "\n")

  lin_model <- lm(reformulate(predictor, response), data = data_train)

  saveRDS(lin_model, "linear_regression.rds")

  r_squared_train <- summary(lin_model)$r.squared

  predictions <- predict(lin_model, newdata = data_test)

  ss_total <- sum((data_test[[response]] - mean(data_test[[response]]))^2)
  ss_res <- sum((data_test[[response]] - predictions)^2)
  r_squared_test <- 1 - (ss_res / ss_total)

  cat("R-squared on train data:", r_squared_train, "\n")
  cat("R-squared on test data:", r_squared_test, "\n")
  print("Model is saved in present directory with the name linear_regression.rds")

  return(list(model = lin_model, r_squared_train = r_squared_train, r_squared_test = r_squared_test, test_predictions = predictions, summary = summary(lin_model)))
}
