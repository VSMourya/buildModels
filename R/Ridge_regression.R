
# Function to fit Ridge Linear Regression and evaluate accuracy
fit_linear_ridge_regression <- function(data, predictors, response, lambda = 1, top_k = NULL) {

  if (!require(glmnet, quietly = TRUE)) {
    install.packages("glmnet")
    library(glmnet)
  } else {
    cat("Package 'glmnet' is already installed.\n")
  }

  data <- na.omit(data)
  if (!is.data.frame(data)) {
    stop("Data must be a dataframe")
  }
  if (!(response %in% names(data))) {
    stop("Predictor or response not found in the dataframe")
  }

  if (!is.null(top_k)) {
    predictors <- select_top_features(data = data, predictors = predictors , response = response, k = top_k)
    print("--------------------------------------------------------------------------------------------------------------------------------------------")
    print("Selected top K Features")
    print(predictors)
  }


  split_result <- train_test_split(data)
  data_train <- split_result$train
  data_test <- split_result$test

  cat("Dimensions of Training Data - Rows:", nrow(data_train), "Columns:", ncol(data_train), "\n")
  cat("Dimensions of Testing Data - Rows:", nrow(data_test), "Columns:", ncol(data_test), "\n")

  print("Starting building model")

  # Prepare data for glmnet
  y_train <- data_train[[response]]
  x_train <- as.matrix(data_train[predictors])

  y_test <- data_test[[response]]
  x_test <- as.matrix(data_test[predictors])

  # Fit Ridge Regression model
  ridge_model <- glmnet(x_train, y_train, alpha = 0, lambda = lambda)

  saveRDS(ridge_model, "ridge_regression.rds")

  # Predict and calculate R-squared for the training data
  predictions_train <- predict(ridge_model, s = lambda, newx = x_train)
  r_squared_train <- 1 - sum((y_train - predictions_train)^2) / sum((y_train - mean(y_train))^2)

  # Predict and calculate R-squared for the test data
  predictions_test <- predict(ridge_model, s = lambda, newx = x_test)
  r_squared_test <- 1 - sum((y_test - predictions_test)^2) / sum((y_test - mean(y_test))^2)

  #print(ridge_model)
  #print(summary(ridge_model))
  cat("R-squared on train data:", r_squared_train, "\n")
  cat("R-squared on test data:", r_squared_test, "\n")
  print("Ridge model is saved in present directory with the name ridge_regression.rds")
  print("--------------------------------------------------------------------------------------------------------------------------------------------")

  return(list(model = ridge_model, r_squared_train = r_squared_train, r_squared_test = r_squared_test, predictors = predictors))
}
