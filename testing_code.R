# Load the buildModels package
library(buildModels)

# Load the dataset
data <- read.csv("QuestionMark.csv")

# Assuming preprocess() and other model functions are properly exported from the package
# Preprocess the data
preprocessed_data <- preprocess(data)

# Define response variable and predictors based on your actual dataset structure
# This is an example assuming 'response' is the column you want to predict
response_var <- "y"  # replace 'response' with the actual column name
predictors <- setdiff(names(preprocessed_data), response_var)

# Fit Linear Regression Model
linear_model_results <- fit_linear_regression(data = preprocessed_data, predictor = predictors, response = response_var)

# Print the summary of the linear model
print(linear_model_results$summary)

# If applicable, fit Logistic Regression Model
# This assumes 'response' is a binary variable suitable for logistic regression
if ("response" %in% names(preprocessed_data)) {
  logistic_model_results <- fit_logistic_regression(preprocessed_data, response_var)
  print(logistic_model_results$summary)
}

# Save models or predictions if needed
saveRDS(linear_model_results$model, "linear_model.rds")
# Example to predict using the model
predictions <- predict(linear_model_results$model, newdata = preprocessed_data)
write.csv(predictions, "predictions.csv", row.names = FALSE)


# ---------------------------------------------------------------------------------------------------------------------------------------------------

# Testing Linear Regression Functions
# lin_reg <- fit_linear_regression(data_preprocessed,predictor = cust_predictors_lin ,response = "MEDV",top_k = 6 )
# lin_ridge_reg <- fit_linear_ridge_regression(data_preprocessed,predictors = cust_predictors_lin ,response = "MEDV",top_k = 6 )
# lin_lasso_reg <- fit_linear_lasso_regression(data_preprocessed,predictors = cust_predictors_lin ,response = "MEDV",top_k = 6 )
# lin_elastic_reg <- fit_linear_elastic_net(data_preprocessed,predictors = cust_predictors_lin ,response = "MEDV" ,top_k = 6)
# lin_svm_reg <- fit_linear_svm(data_preprocessed,predictors = cust_predictors_lin ,response = "MEDV" ,top_k = 6)
# lin_rf_reg <- fit_random_forest(data_preprocessed,predictors = cust_predictors_lin ,response = "MEDV" ,top_k = 6)
# lin_bt_reg <- fit_boosted_trees(data_preprocessed,predictors = cust_predictors_lin ,response = "MEDV" ,top_k = 6)
# models_linear <- c("linear_regression", "ridge_regression", "lasso_regression", "elastic_net", "svm_regression", "random_forest", "boosted_trees")
# ensemble_predictions_linear <- ensemble_learning(data_preprocessed, cust_predictors_lin, "MEDV", models_linear)

# Testing Logistic Regression Functions
# log_reg <- fit_logistic_regression(data_processed_log,predictor = cust_predictors_log ,response = "Class" ,top_k = 6)
# log_ridge_reg <- fit_logistic_ridge(data_processed_log,predictor = cust_predictors_log ,response = "Class" ,top_k = 6)
# log_lasso_reg <- fit_logistic_lasso(data_processed_log,predictor = cust_predictors_log ,response = "Class" ,top_k = 6)
# log_elastic_reg <- fit_logistic_elastic_net(data_processed_log,predictor = cust_predictors_log ,response = "Class" ,top_k = 6)
# log_svm_reg <- fit_logistic_svm(data_processed_log,predictor = cust_predictors_log ,response = "Class" ,top_k = 6)
# log_rf_reg <- fit_random_forest_binary(data_processed_log,predictor = cust_predictors_log ,response = "Class" ,top_k = 6)
# log_gt_reg <- fit_boosted_trees_binary(data_processed_log,predictor = cust_predictors_log ,response = "Class" ,top_k = 6)
# models_list_bianry <- c("linear_regression", "ridge_regression", "lasso_regression", "elastic_net", "svm_regression", "random_forest")
# ensemble_predictions_linear <- ensemble_learning_binary(data_processed_log,predictor = cust_predictors_log ,response = "Class", models = models_list_binary)



