# Load necessary libraries
library(gbm)
library(xgboost)
library(glmnet)
library(e1071)
library(randomForest)

# Example dataset
data(iris)
iris$Species <- as.numeric(iris$Species == "setosa")  # Assuming binary outcome for logistic models

# Define predictors and response
predictors <- names(iris)[1:4]
response <- "Species"

# Split the data into training and testing sets
set.seed(123)  # for reproducibility
train_indices <- sample(1:nrow(iris), 0.7 * nrow(iris))
train_data <- iris[train_indices, ]
test_data <- iris[-train_indices, ]

# 1. Test Linear Regression
lin_reg_results <- fit_linear_regression(train_data, predictors, response)
print(lin_reg_results)

# 2. Test Logistic Regression with Ridge
ridge_log_results <- fit_logistic_ridge(train_data, predictors, response, lambda = 0.1)
print(ridge_log_results)

# 3. Test SVM for Linear Regression
svm_lin_results <- fit_linear_svm(train_data, predictors, response)
print(svm_lin_results)

# 4. Test SVM for Logistic Regression
svm_log_results <- fit_logistic_svm(train_data, predictors, response)
print(svm_log_results)

# 5. Test Boosted Trees Linear Regression
# boosted_lin_results <- fit_boosted_trees(train_data, predictors, response)
# print(boosted_lin_results)

# 6. Test Boosted Trees Logistic Regression
boosted_log_results <- fit_boosted_trees_binary(train_data, predictors, response)
print(boosted_log_results)

# 7. Test Elastic Net Linear Regression
elastic_lin_results <- fit_linear_elastic_net(train_data, predictors, response, lambda = 0.1)
print(elastic_lin_results)

# 8. Test Elastic Net Logistic Regression
elastic_log_results <- fit_logistic_elastic_net(train_data, predictors, response, lambda = 0.1)
print(elastic_log_results)

# 9. Test Ensemble Model
model_types <- c("linear_regression", "ridge_regression", "lasso_regression", "elastic_net",
                 "svm_regression", "random_forest", "boosted_trees")
ensemble_results <- ensemble_learning(train_data, predictors, response, model_types)
print(ensemble_results)
