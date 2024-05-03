
#' Fit a Random Forest Model and Evaluate Accuracy
#'
#' This function fits a Random Forest regression model using the `randomForest` package.
#' It handles installation and loading of the necessary package, checks if the response
#' variable is numeric, and optionally selects the top K important predictors. The dataset
#' is split into training and testing sets, a model is fitted, and RMSE for both sets are
#' calculated. The model is saved to the current directory.
#'
#' @param data A data frame containing the training dataset.
#' @param predictors A character vector of predictor variable names in the dataset.
#' @param response A string naming the response variable in the dataset.
#' @param top_k Optional; an integer specifying the number of top features to select based on importance.
#'
#' @return A list containing the random forest model, training RMSE, testing RMSE, and the predictors used.
#' @import randomForest
#' @examples
#' data(iris)
#' result <- fit_random_forest(data = iris, predictors = c("Sepal.Length", "Sepal.Width"), response = "Petal.Length")
#' print(result)
#'
#' @export

# Function to fit Random Forest Linear Regression and evaluate accuracy
fit_random_forest <- function(data, predictors, response, top_k = NULL) {
  # Ensure the 'randomForest' package is installed and loaded
  if (!require(randomForest, quietly = TRUE)) {
    install.packages("randomForest")
    library(randomForest)
  }

  # Verify the response variable is numeric
  if (!is.numeric(data[[response]])) {
    stop("Response variable must be numeric.")
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

  # Save and evaluate the model
  saveRDS(rf_model, "random_forest.rds")
  train_rmse <- sqrt(mean((predict(rf_model, data_train) - data_train[[response]])^2))
  test_rmse <- sqrt(mean((predict(rf_model, data_test) - data_test[[response]])^2))

  cat(sprintf("Training RMSE: %.4f, Testing RMSE: %.4f\n", train_rmse, test_rmse))
  cat("Random Forest model saved as 'random_forest.rds'\n")

  return(list(model = rf_model, rmse_train = train_rmse, rmse_test = test_rmse, predictors = predictors))
}
