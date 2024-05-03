#' Fit a Boosted Trees Linear Regression Model and Evaluate RMSE
#'
#' This function fits a Boosted Trees model using the `gbm` package. It checks the data integrity,
#' optionally selects the top K features, fits the model, and calculates RMSE on both training and
#' testing datasets. The model is saved to the current directory.
#'
#' @param data A dataframe containing the dataset.
#' @param predictors A character vector of predictor variable names.
#' @param response A string indicating the response variable name.
#' @param top_k Optional; an integer for selecting top K features based on importance.
#'
#' @return A list containing the fitted model, training RMSE, testing RMSE, and the predictors used.
#' @import gbm
#' @examples
#' data(iris)
#' iris$Species <- as.numeric(iris$Species)
#' result <- fit_boosted_trees(data = iris, predictors = c("Sepal.Length", "Sepal.Width"), response = "Species")
#' print(result)
#' @export

fit_boosted_trees <- function(data, predictors, response, top_k = NULL) {
  if (!require(gbm, quietly = TRUE)) {
    install.packages("gbm")
    library(gbm)
  }

  data <- na.omit(data)
  if (!is.data.frame(data)) stop("Data must be a dataframe")
  if (!(response %in% names(data))) stop("Response variable not found in the dataframe")
  if (!all(predictors %in% names(data))) stop("Some predictors not found in the dataframe")
  if (nrow(data) == 0) stop("No data available after removing NA values")
  if (!is.numeric(data[[response]])) stop("Response variable must be numeric")

  if (!is.null(top_k)) {
    predictors <- select_top_features(data = data, predictors = predictors, response = response, k = top_k)
    cat("Selected top K Features:\n", paste(predictors, collapse=", "), "\n")
  }

  split_result <- train_test_split(data)
  data_train <- split_result$train
  data_test <- split_result$test

  boosted_model <- gbm(as.formula(paste(response, "~", paste(predictors, collapse = "+"))),
                       data = data_train, distribution = "gaussian", n.trees = 100,
                       interaction.depth = 4, shrinkage = 0.01, bag.fraction = 0.5, cv.folds = 5)

  saveRDS(boosted_model, "boosted_trees.rds")
  train_predictions <- predict(boosted_model, data_train, n.trees = 100)
  train_rmse <- sqrt(mean((train_predictions - data_train[[response]])^2))
  test_predictions <- predict(boosted_model, data_test, n.trees = 100)
  test_rmse <- sqrt(mean((test_predictions - data_test[[response]])^2))

  cat("RMSE on training data:", train_rmse, "\n")
  cat("RMSE on test data:", test_rmse, "\n")
  cat("Boosted Trees model is saved in the present directory with the name boosted_trees.rds\n")

  return(list(model = boosted_model, rmse_train = train_rmse, rmse_test = test_rmse, predictors = predictors))
}
