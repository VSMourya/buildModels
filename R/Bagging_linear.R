#' Bagging Model for Various Regression Types
#'
#' This function implements bagging (bootstrap aggregating) for different types of regression models
#' including linear, logistic, ridge, lasso, and elastic net. It trains specified model types on bootstrapped
#' samples and aggregates the predictions, calculating RMSE and variable importance based on the coefficients
#' from each model iteration.
#'
#' @param data A dataframe containing the dataset.
#' @param predictors A character vector of predictor variable names.
#' @param response A string indicating the response variable name.
#' @param model_type A string specifying the type of model to fit: linear_regression, ridge_regression,
#'        lasso_regression, or elastic_net.
#' @param R An integer specifying the number of bootstrap replicates.
#'
#' @return A list containing aggregated predictions, RMSE, coefficient summaries, and variable importance.
#' @importFrom stats predict
#' @examples
#' data(iris)
#' iris$Species <- as.numeric(iris$Species)
#' result <- bagging_linear(data = iris, predictors = c("Sepal.Length", "Sepal.Width"), response = "Species", model_type = "linear_regression", R = 100)
#' print(result)
#' @export

bagging_linear <- function(data, predictors, response, model_type, R = 100) {
  predictions <- matrix(NA, nrow = nrow(data), ncol = R)
  variable_importance <- rep(0, length(predictors))
  coef_summaries <- list()

  for (i in 1:R) {
    indices <- sample(nrow(data), size = as.integer(nrow(data) * 0.9), replace = TRUE)
    data_boot <- data[indices, ]
    cat(sprintf("Dimensions of Bootstrapped Data - Iteration %d: Rows: %d, Columns: %d\n", i, nrow(data_boot), ncol(data_boot)))

    model <- switch(model_type,
                    "linear_regression" = fit_linear_regression(data_boot, predictors, response),
                    "ridge_regression" = fit_linear_ridge_regression(data_boot, predictors, response),
                    "lasso_regression" = fit_linear_lasso_regression(data_boot, predictors, response),
                    "elastic_net" = fit_linear_elastic_net(data_boot, predictors, response),
                    stop("Invalid model type"))

    df_pred <- as.matrix(data[, predictors])
    predictions[, i] <- predict(model$model, newx = df_pred, type = "response")
    best_coefs <- coef(model$model, s = "lambda.min")
    coef_vector <- as.numeric(best_coefs[names(best_coefs) %in% predictors])
    names(coef_vector) <- predictors
    coef_summaries[[i]] <- coef_vector
  }

  feature_weights <- matrix(unlist(coef_summaries), nrow = R, byrow = TRUE)
  variable_importance <- colMeans(abs(feature_weights), na.rm = TRUE)
  names(variable_importance) <- predictors

  final_predictions <- rowMeans(predictions, na.rm = TRUE)
  squared_errors <- (data[[response]] - final_predictions)^2
  rmse <- sqrt(mean(squared_errors))

  cat("RMSE for bagging is:", rmse, "\n")

  return(list(predictions = final_predictions, rmse = rmse, coef_summaries = coef_summaries, variable_importance = variable_importance))
}
