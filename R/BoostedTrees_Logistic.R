#' Fit a Boosted Trees Logistic Regression Model and Evaluate Accuracy
#'
#' This function fits a Boosted Trees model using the `xgboost` package for binary classification.
#' It preprocesses data, handles NA values, optionally selects top K features, and evaluates the model's
#' accuracy on both training and testing sets. The model is saved to the current directory.
#'
#' @param data A dataframe containing the dataset.
#' @param predictors A character vector of predictor variable names.
#' @param response A string indicating the response variable name.
#' @param top_k Optional; an integer for selecting top K features based on importance.
#'
#' @return A list containing the fitted model, training accuracy, testing accuracy, and the predictors used.
#' @import xgboost
#' @examples
#' data(iris)
#' iris$Species <- ifelse(iris$Species == "setosa", 1, 0)
#' result <- fit_boosted_trees_binary(data = iris, predictors = c("Sepal.Length", "Sepal.Width"), response = "Species")
#' print(result)
#' @export

fit_boosted_trees_binary <- function(data, predictors, response, top_k = NULL) {
  if (!require(xgboost, quietly = TRUE)) {
    install.packages("xgboost")
    library(xgboost)
  }

  data <- na.omit(data)
  if (!is.data.frame(data)) stop("Data must be a dataframe")
  if (!(response %in% names(data))) stop("Response variable not found in the dataframe")
  if (nrow(data) == 0) stop("No data available after removing NA values")
  if (!is.factor(data[[response]]) || length(levels(data[[response]])) != 2) stop("Response variable must be binary (factor with 2 levels)")

  if (!is.null(top_k)) {
    predictors <- select_top_features(data = data, predictors = predictors, response = response, k = top_k)
    cat("Selected top K Features:\n", paste(predictors, collapse=", "), "\n")
  }

  split_result <- train_test_split(data)
  data_train <- split_result$train
  data_test <- split_result$test

  dtrain <- xgb.DMatrix(data = as.matrix(data_train[predictors]), label = data_train[[response]])
  dtest <- xgb.DMatrix(data = as.matrix(data_test[predictors]), label = data_test[[response]])

  params <- list(
    objective = "binary:logistic",
    eval_metric = "error",
    eta = 0.01,
    max_depth = 4,
    subsample = 0.5
  )

  boosted_model <- xgboost(params = params, data = dtrain, nrounds = 100, verbose = 0)
  xgb.save(boosted_model, "boosted_trees_binary.model")

  train_predictions <- predict(boosted_model, dtrain)
  train_accuracy <- mean((train_predictions > 0.5) == data_train[[response]])
  test_predictions <- predict(boosted_model, dtest)
  test_accuracy <- mean((test_predictions > 0.5) == data_test[[response]])

  cat("Accuracy on train data:", train_accuracy, "\n")
  cat("Accuracy on test data:", test_accuracy, "\n")
  cat("Boosted Trees model is saved under 'boosted_trees_binary.model'\n")

  return(list(model = boosted_model, accuracy_train = train_accuracy, accuracy_test = test_accuracy, predictors = predictors))
}
