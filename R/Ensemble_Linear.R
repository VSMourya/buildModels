#' Ensemble Model for Linear Regression
#'
#' This function creates an ensemble model for linear regression using various model types.
#' It averages predictions from different regression models to compute a combined RMSE.
#'
#' @param data A dataframe containing the dataset.
#' @param predictors A character vector of predictor variable names.
#' @param response A string indicating the response variable name.
#' @param models A list of model types to include in the ensemble.
#'
#' @return A vector of combined predictions from the ensemble model.
#' @examples
#' data(iris)
#' iris$Species <- as.numeric(iris$Species)
#' models <- c("linear_regression", "ridge_regression", "lasso_regression", "elastic_net", "svm_regression", "random_forest", "boosted_trees")
#' predictions <- ensemble_learning(data = iris, predictors = c("Sepal.Length", "Sepal.Width"), response = "Species", models = models)
#' print(predictions)
#' @export

ensemble_learning <- function(data, predictors, response, models) {
  predictions <- list()
  for (model_type in models) {
    switch(model_type,
           "linear_regression" = {
             lin_reg <- fit_linear_regression(data, predictors, response)
             predictions[[model_type]] <- lin_reg$test_predictions
           },
           "ridge_regression" = {
             ridge_reg <- fit_linear_ridge_regression(data, predictors, response)
             predictions[[model_type]] <- predict(ridge_reg$model, newx = as.matrix(data[predictors]))
           },
           "lasso_regression" = {
             lasso_reg <- fit_linear_lasso_regression(data, predictors, response)
             predictions[[model_type]] <- predict(lasso_reg$model, newx = as.matrix(data[predictors]))
           },
           "elastic_net" = {
             elastic_reg <- fit_linear_elastic_net(data, predictors, response)
             predictions[[model_type]] <- predict(elastic_reg$model, newx = as.matrix(data[predictors]))
           },
           "svm_regression" = {
             svm_reg <- fit_linear_svm(data, predictors, response)
             predictions[[model_type]] <- predict(svm_reg$model, newdata = data)
           },
           "random_forest" = {
             rf_reg <- fit_random_forest(data, predictors, response)
             predictions[[model_type]] <- predict(rf_reg$model, newdata = data)
           },
           "boosted_trees" = {
             bt_reg <- fit_boosted_trees(data, predictors, response)
             predictions[[model_type]] <- predict(bt_reg$model, newdata = data, n.trees = 100)
           },
           {
             print(paste("Model type", model_type, "not recognized"))
           }
    )
  }

  combined_predictions <- rowMeans(do.call(cbind, predictions))
  squared_errors <- (data[[response]] - combined_predictions)^2
  mean_squared_error <- mean(squared_errors)
  rmse <- sqrt(mean_squared_error)

  cat("RMSE for ensemble model is:", rmse, "\n")
  print("--------------------------------------------------------------------------------------------------------------------------------------------")
  return(combined_predictions)
}
