#' Bagging Model for Logistic Regression
#'
#' This function implements bagging (bootstrap aggregating) for logistic regression and related models.
#' It trains the specified model on bootstrapped samples of the data and aggregates the predictions.
#' It calculates combined accuracy and variable importance based on the coefficients from each model.
#'
#' @param data A dataframe containing the dataset.
#' @param predictors A character vector of predictor variable names.
#' @param response A string indicating the response variable name.
#' @param model_type A string specifying the type of model to fit: logistic_regression, ridge_regression,
#'        lasso_regression, or elastic_net.
#' @param R An integer specifying the number of bootstrap replicates.
#'
#' @return A list containing combined predictions, overall accuracy, coefficient summaries, and variable weights.
#' @importFrom stats predict
#' @importFrom utils head
#' @examples
#' data(iris)
#' iris$Species <- ifelse(iris$Species == "setosa", 1, 0)
#' result <- bagging_logistic(data = iris, predictors = c("Sepal.Length", "Sepal.Width"), response = "Species", model_type = "logistic_regression", R = 100)
#' print(result)
#' @export

bagging_logistic <- function(data, predictors, response, model_type, R = 100) {
  predictions <- matrix(NA, nrow = nrow(data), ncol = R)
  coef_summaries <- list()  # Store coefficient values from each model iteration

  for (i in 1:R) {
    # Generate bootstrap sample
    indices <- sample(nrow(data), size = as.integer(nrow(data) * 0.9), replace = TRUE)
    data_boot <- data[indices, ]
    cat(sprintf("Dimensions of Bootstrapped Data - Iteration %d: Rows: %d, Columns: %d\n", i, nrow(data_boot), ncol(data_boot)))

    # Train the model on the bootstrap sample
    if (model_type == "logistic_regression") {
      model <- fit_logistic_regression(data_boot, predictors, response)
    } else if (model_type == "ridge_regression") {
      model <- fit_logistic_ridge(data_boot, predictors, response)
    } else if (model_type == "lasso_regression") {
      model <- fit_logistic_lasso(data_boot, predictors, response)
    } else if (model_type == "elastic_net") {
      model <- fit_logistic_elastic_net(data_boot, predictors, response)
    } else {
      stop("Invalid model type")
    }

    # Coefficients extraction and predictions
    df_pred <- data[, -which(names(data) == response)]
    predictions[, i] <- predict(model$model, newdata = df_pred, type = "response")
    coef_vector <- coef(model$model, s = "lambda.min")
    coef_summaries[[i]] <- as.list(coef_vector)
  }

  # Aggregate predictions and calculate accuracy
  final_predictions <- rowMeans(predictions, na.rm = TRUE)
  combined_classes <- ifelse(final_predictions > 0.5, 1, 0)
  combined_accuracy <- mean(combined_classes == data[[response]])

  cat("Accuracy for bagging is:", combined_accuracy, "\n")
  print("--------------------------------------------------------------------------------------------------------------------------------------------")

  # Calculate variable importance
  feature_weights <- sapply(coef_summaries, function(x) abs(unlist(x[names(x) %in% predictors])))
  variable_importance <- colMeans(feature_weights, na.rm = TRUE)

  return(list(predictions = combined_classes, accuracy = combined_accuracy, coef_summaries = coef_summaries, variable_weights = variable_importance))
}
