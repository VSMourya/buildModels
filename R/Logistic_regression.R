#' Fit Logistic Regression and Evaluate Accuracy
#'
#' This function fits a logistic regression model using the glm function and evaluates
#' its accuracy on both training and testing datasets. It supports feature selection by selecting
#' the top k predictors based on importance.
#'
#' @param data A data frame containing the predictors and response variable.
#' @param predictor A character vector of predictor variable names.
#' @param response The name of the response variable.
#' @param top_k The number of top k predictors to select based on importance; if NULL, uses all predictors.
#' @return A list containing the model, testing accuracy, predictions, and model summary.
#' @examples
#' \dontrun{
#'   data <- data.frame(matrix(rnorm(100 * 10), ncol = 10))
#'   colnames(data) <- paste0("X", 1:10)
#'   data$Y <- sample(0:1, 100, replace = TRUE)
#'   result <- fit_logistic_regression(data, predictors = paste0("X", 1:10), response = "Y")
#' }
#' @export
fit_logistic_regression <- function(data, predictor, response, top_k = NULL) {
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

  log_model <- glm(reformulate(predictor, response), data = data_train, family = binomial())

  saveRDS(log_model, "logistic_regression.rds")

  predicted_probabilities_train <- predict(log_model, data_train, type = "response")
  predicted_classes_train <- ifelse(predicted_probabilities_train > 0.5, 1, 0)
  correct_predictions_train <- mean(predicted_classes_train == data_train[[response]])

  predicted_probabilities_test <- predict(log_model, data_test, type = "response")
  predicted_classes_test <- ifelse(predicted_probabilities_test > 0.5, 1, 0)
  correct_predictions_test <- mean(predicted_classes_test == data_test[[response]])

  cat("Accuracy on train data:", correct_predictions_train, "\n")
  cat("Accuracy on test data:", correct_predictions_test, "\n")
  print("Model is saved in present directory with the name logistic_regression")

  return(list(model = log_model, accuracy_train = correct_predictions_train, accuracy_test = correct_predictions_test, test_predictions = predicted_classes_test, summary = summary(log_model)))
}
