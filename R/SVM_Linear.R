#' Fit a Linear SVM Regression Model and Evaluate RMSE
#'
#' This function fits a Linear SVM Regression model using the `e1071` package.
#' It ensures the package is installed and loaded, checks the numeric nature of the response variable,
#' optionally selects top features, fits the model using a linear kernel, predicts, calculates RMSE,
#' and saves the model to the working directory.
#'
#' @param data A dataframe containing the dataset.
#' @param predictors A character vector of predictor variable names.
#' @param response A string indicating the response variable name.
#' @param top_k Optional; an integer for selecting top K features based on importance.
#'
#' @return A list containing the fitted SVM model, training RMSE, testing RMSE, and the predictors used.
#' @import e1071
#' @examples
#' data(iris)
#' iris$Species <- as.numeric(iris$Species)
#' result <- fit_linear_svm(data = iris, predictors = c("Sepal.Length", "Sepal.Width"), response = "Species")
#' print(result)
#' @export

fit_linear_svm <- function(data, predictors, response, top_k = NULL) {
  # Ensure the 'e1071' package is installed and loaded
  if (!require(e1071, quietly = TRUE)) {
    install.packages("e1071")
    library(e1071)
  }

  # Validate response variable
  if (!is.numeric(data[[response]])) {
    stop("Response variable must be numeric.")
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

  # Fit Linear SVM model
  formula_svm <- as.formula(paste(response, "~", paste(predictors, collapse = "+")))
  svm_model <- svm(formula_svm, data = data_train, kernel = "linear")
  saveRDS(svm_model, "linear_svm.rds")

  # Predict and calculate RMSE for training data
  train_predictions <- predict(svm_model, data_train)
  train_rmse <- sqrt(mean((train_predictions - data_train[[response]])^2))

  # Predict and calculate RMSE for testing data
  test_predictions <- predict(svm_model, data_test)
  test_rmse <- sqrt(mean((test_predictions - data_test[[response]])^2))

  cat(sprintf("Training RMSE: %.4f, Testing RMSE: %.4f\n", train_rmse, test_rmse))
  cat("Linear SVM model saved as 'linear_svm.rds'\n")

  return(list(model = svm_model, rmse_train = train_rmse, rmse_test = test_rmse, predictors = predictors))
}
