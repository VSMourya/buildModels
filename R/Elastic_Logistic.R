#' Fit a Logistic Elastic Net Regression Model and Evaluate Accuracy
#'
#' This function fits an Elastic Net Logistic Regression model using the `glmnet` package for binary classification.
#' It preprocesses data, optionally selects the top K features, fits the model, evaluates accuracy on the testing dataset,
#' and saves the model to the current directory.
#'
#' @param data A dataframe containing the dataset.
#' @param predictors A character vector of predictor variable names.
#' @param response A string indicating the response variable name.
#' @param lambda Regularization parameter; default is 1.
#' @param top_k Optional; an integer for selecting top K features based on importance.
#'
#' @return A list containing the fitted model, accuracy on test data, and the predictors used.
#' @import glmnet
#' @examples
#' data(iris)
#' iris$Species <- ifelse(iris$Species == "setosa", 1, 0)
#' result <- fit_logistic_elastic_net(data = iris, predictors = c("Sepal.Length", "Sepal.Width"), response = "Species")
#' print(result)
#' @export

fit_logistic_elastic_net <- function(data, predictors, response, lambda = 1, top_k = NULL) {
  if (!requireNamespace("glmnet", quietly = TRUE)) {
    install.packages("glmnet")
    library(glmnet)
  }

  data <- na.omit(data)
  if (!is.data.frame(data)) stop("Data must be a dataframe.")
  if (!(response %in% names(data))) stop("Response variable not found in the dataframe.")
  if (!all(predictors %in% names(data))) stop("Some predictors not found in the dataframe.")
  if (lambda < 0) stop("Lambda must be non-negative.")
  if (nrow(data) == 0) stop("No data available after removing NA values.")

  if (!is.null(top_k) && top_k > 0) {
    predictors <- select_top_features(data = data, predictors = predictors, response = response, k = top_k)
    print("Selected top K Features:")
    print(predictors)
  }

  split_result <- train_test_split(data)
  data_train <- split_result$train
  data_test <- split_result$test

  x_train <- model.matrix(reformulate(predictors, response), data_train)[, -1]
  y_train <- as.numeric(data_train[[response]]) - 1  # Ensure binary response is numeric 0/1

  elastic_model <- glmnet(x_train, y_train, family = "binomial", alpha = 0.5, lambda = lambda)
  saveRDS(elastic_model, "elastic_net_logistic.rds")

  x_test <- model.matrix(reformulate(predictors, response), data_test)[, -1]
  predictions_test <- predict(elastic_model, newx = x_test, s = lambda, type = "response")
  correct_predictions_test <- mean((predictions_test > 0.5) == (data_test[[response]] - 1))

  cat("Accuracy on test data: ", correct_predictions_test, "\n")
  cat("Elastic Net Logistic Regression model is saved under 'elastic_net_logistic.rds'\n")

  return(list(model = elastic_model, accuracy_test = correct_predictions_test, predictors = predictors))
}
