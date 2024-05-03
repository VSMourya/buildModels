#' Ensemble Learning for Linear Regression Models
#'
#' This function combines predictions from multiple regression models to form a more robust prediction using ensemble techniques. It calculates the mean squared error (RMSE) of the combined predictions.
#'
#' @param data A data frame containing the predictors and the response variable.
#' @param predictors A character vector of predictor variable names in the data.
#' @param response The name of the response variable in the data.
#' @param models A character vector listing the types of models to include in the ensemble. Recognized types are: "linear_regression", "ridge_regression", "lasso_regression", "elastic_net", "svm_regression", "random_forest", and "boosted_trees".
#'
#' @return A list containing the combined predictions and the RMSE of these predictions.
#'
#' @examples
#' \dontrun{
#'   data <- read.csv("path/to/your/data.csv")
#'   predictors <- c("predictor1", "predictor2", "predictor3")
#'   response <- "outcome"
#'   models <- c("linear_regression", "ridge_regression", "lasso_regression")
#'   results <- ensemble_learning(data, predictors, response, models)
#'   print(results$combined_predictions)
#'   print(results$RMSE)
#' }
#'
#' @export
ensemble_learning <- function(data, predictors, response, models) {
  if (!is.data.frame(data)) {
    stop("Data must be a dataframe.")
  }
  if (!(response %in% names(data))) {
    stop("Response variable not found in the dataframe.")
  }
  if (!all(predictors %in% names(data))) {
    stop("Some predictors not found in the dataframe.")
  }
  if (length(models) == 0) {
    stop("No models specified for the ensemble.")
  }

  predictions <- list()
  for (model_type in models) {
    tryCatch({
      if (model_type == "linear_regression") {
        model_results <- fit_linear_regression(data, predictors, response)
      } else if (model_type == "ridge_regression") {
        model_results <- fit_linear_ridge_regression(data, predictors, response)
      } else if (model_type == "lasso_regression") {
        model_results <- fit_linear_lasso_regression(data, predictors, response)
      } else if (model_type == "elastic_net") {
        model_results <- fit_linear_elastic_net(data, predictors, response)
      } else if (model_type == "svm_regression") {
        model_results <- fit_linear_svm(data, predictors, response)
      } else if (model_type == "random_forest") {
        model_results <- fit_random_forest(data, predictors, response)
      } else if (model_type == "boosted_trees") {
        model_results <- fit_boosted_trees(data, predictors, response)
      } else {
        warning(paste("Model type", model_type, "not recognized"))
        next
      }
      predictions[[model_type]] <- predict(model_results$model, newdata = data)
    }, error = function(e) {
      warning(paste("Failed to process model", model_type, ":", e$message))
    })
  }

  if (length(predictions) == 0) {
    stop("No valid model predictions were generated.")
  }

  combined_predictions <- rowMeans(do.call(cbind, predictions), na.rm = TRUE)
  squared_errors <- (data[[response]] - combined_predictions)^2
  mean_squared_error <- mean(squared_errors, na.rm = TRUE)
  rmse <- sqrt(mean_squared_error)

  cat("RMSE for ensemble model is:", rmse, "\n")

  return(list(combined_predictions = combined_predictions, RMSE = rmse))
}
