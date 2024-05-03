# This project is going to be a lazy predictor.
# It is abled to be used both for Logistic Regression and Linear Regression.
# Methods to be used are as follows:
# 1. Linear/Logistic Regression
# 2. Ridge Regression
# 3. Lasso Regression
# 4. Elastic Net
# 5. Support Vector Machine, Random Forest, Boosted Trees
# Implement Forward Selection.
# Perform Bagging - Sampling with Replacement - Write own function.
# Final model will be average from different models generated using Bagging.
# Implement Ensemble model to fit more than one model - Write own code.
# Naive weight for predictors - What is this ?

#' Preprocess the data by handling NA values, outliers, and encoding categorical variables
#'
#' @param df A data frame to be processed.
#' @param Nan_Fill Logical indicating if NA values should be filled.
#' @param print_info Logical indicating if the function should print detailed information.
#' @return A preprocessed data frame.
#'
preprocess <- function(df, Nan_Fill = TRUE, print_info = TRUE) {
  if (print_info) {
    print(paste("Number of rows in input data frame:", nrow(df)))
    print(paste("Number of Columns in input data frame:", ncol(df)))
    print("--------------------------------------------------------------")
    print("Count of NaN Values in given data:")
    nan_counts <- sapply(df, function(x) sum(is.na(x)))
    print(nan_counts)
    print("--------------------------------------------------------------")
  }

  # Ensure data is properly typed before further processing
  df <- convert_if_ordinal(df)  # Call this before split_data if you include the function

  result <- split_data(df)
  numeric_df <- result$numeric
  ordinal_df <- result$ordinal
  categorical_df <- result$categorical

  if (print_info) {
    print("Initial Data Split:")
    print(head(numeric_df))
    print(head(ordinal_df))
    print(head(categorical_df))
    print("--------------------------------------------------------------")
  }

  num_nonan <- fill_na_values(numeric_df)
  ord_nonan <- fill_na_values(ordinal_df, ord = TRUE)
  cat_nonan <- fill_na_values(categorical_df)

  num_nonan <- clip_outliers_in_dataframe(num_nonan)

  cat_encoded <- convert_categoricals(cat_nonan)

  data_processed <- cbind(cat_encoded, num_nonan, ordinal_df)

  if (print_info) {
    print("Final Preprocessed Data:")
    print(head(data_processed))
  }
  return(data_processed)
}

#' Test preprocessing on new data using stored fill values information
#'
#' @param df A data frame to be processed.
#' @return A data frame with filled NA values.
preprocess_test <- function(df) {
  print("Testing Preprocessing:")
  return(fill_na_values_test(df))
}

#' Split data into numeric, ordinal, and categorical dataframes based on defined criteria
#'
#' @param df Data frame to split.
#' @param ordinal_threshold Threshold to classify a numeric variable as ordinal.
#' @return A list containing split data frames.
split_data <- function(df, ordinal_threshold = 0.05) {
  is_ordinal <- function(x) {
    integral <- all(x == floor(x))
    unique_ratio <- length(unique(x)) / length(x)
    return(integral && unique_ratio <= ordinal_threshold)
  }

  numeric_cols <- sapply(df, is.numeric)
  ordinal_cols <- sapply(df, is_ordinal)
  categorical_cols <- !(numeric_cols | ordinal_cols)

  numeric_df <- df[, numeric_cols, drop = FALSE]
  ordinal_df <- df[, ordinal_cols, drop = FALSE]
  categorical_df <- df[, categorical_cols, drop = FALSE]

  return(list(numeric = numeric_df, ordinal = ordinal_df, categorical = categorical_df))
}

#' Select top features based on variable importance using a random forest model
#'
#' @param data Data frame containing the predictor and response variables.
#' @param predictors Names of predictor variables.
#' @param response Name of the response variable.
#' @param k Number of top features to return.
#' @return Names of top k important features.
select_top_features <- function(data, predictors, response, k = 10) {
  if (!require(randomForest, quietly = TRUE)) {
    install.packages("randomForest")
    library(randomForest)
  }
  if (!require(dplyr, quietly = TRUE)) {
    install.packages("dplyr")
    library(dplyr)
  }

  rf_model <- randomForest(as.formula(paste(response, "~", paste(predictors, collapse = "+"))), data = data)
  var_importance <- importance(rf_model)

  df <- data.frame(Feature = predictors, Importance = var_importance)
  sorted_df <- arrange(df, desc(Importance))

  return(sorted_df$Feature[1:k])
}


# Function to induce random missingness in the dataframe
random_nan <- function(column) {
  n <- length(column)
  num_nans <- sample(round(0.05 * n):round(0.10 * n), 1)
  nan_indices <- sample(1:n, num_nans)
  column[nan_indices] <- NA
  return(column)
}

# Function to fill NaN values in Dataframe
fill_na_values <- function(df, ord = FALSE) {
  fill_values_info <- data.frame(Column_Name = character(), Fill_Value = numeric(), stringsAsFactors = FALSE)
  calculate_mode <- function(x) {
    ux <- unique(x)
    ux[which.max(tabulate(match(x, ux)))]
  }

  for (col_name in names(df)) {
    if (is.numeric(df[[col_name]]) || ord) {
      mode_value <- calculate_mode(df[[col_name]][!is.na(df[[col_name]])])
      df[[col_name]][is.na(df[[col_name]])] <- mode_value
      fill_values_info <- rbind(fill_values_info, data.frame(Column_Name = col_name, Fill_Value = mode_value))
    } else {
      mean_value <- mean(df[[col_name]], na.rm = TRUE)
      df[[col_name]][is.na(df[[col_name]])] <- mean_value
      fill_values_info <- rbind(fill_values_info, data.frame(Column_Name = col_name, Fill_Value = mean_value))
    }
  }
  write.csv(fill_values_info, "fill_values_info.csv", row.names = FALSE)
  return(df)
}

# Function to apply the fill_na_values function to new data using previously stored values
fill_na_values_test <- function(test_df, fill_values_info_path = "fill_values_info.csv") {
  fill_values_info <- read.csv(fill_values_info_path, stringsAsFactors = FALSE)
  for (col_name in names(test_df)) {
    if (col_name %in% fill_values_info$Column_Name) {
      fill_value <- fill_values_info$Fill_Value[fill_values_info$Column_Name == col_name]
      test_df[[col_name]][is.na(test_df[[col_name]])] <- fill_value
    }
  }
  return(test_df)
}

# Function to convert factors to numeric if they are ordinal and numeric checks are required
convert_if_ordinal <- function(df) {
  for (col_name in names(df)) {
    if (is.factor(df[[col_name]]) && is_ordinal(df[[col_name]])) {
      df[[col_name]] <- as.numeric(as.character(df[[col_name]]))
    }
  }
  return(df)
}

# Function to split data into numeric, ordinal, and categorical dataframes
split_data <- function(df, ordinal_threshold = 0.05) {
  is_ordinal <- function(x) {
    if(is.numeric(x)) {
      integral <- all(x == floor(x))
      unique_ratio <- length(unique(x)) / length(x)
      return(integral && unique_ratio <= ordinal_threshold)
    }
    return(FALSE)
  }

  numeric_cols <- sapply(df, is.numeric)
  ordinal_cols <- sapply(df, is_ordinal)
  categorical_cols <- !(numeric_cols | ordinal_cols)

  numeric_df <- df[, numeric_cols, drop = FALSE]
  ordinal_df <- df[, ordinal_cols, drop = FALSE]
  categorical_df <- df[, categorical_cols, drop = FALSE]

  return(list(numeric = numeric_df, ordinal = ordinal_df, categorical = categorical_df))
}

# Function to clip outliers in a dataframe
clip_outliers_in_dataframe <- function(df, probs = c(0.05, 0.95)) {
  bounds_info <- data.frame(Column_Name = character(), Lower_Bound = numeric(), Upper_Bound = numeric(), stringsAsFactors = FALSE)
  for (col_name in names(df)) {
    if (is.numeric(df[[col_name]])) {
      lower_bound <- quantile(df[[col_name]], probs[1], na.rm = TRUE)
      upper_bound <- quantile(df[[col_name]], probs[2], na.rm = TRUE)
      df[[col_name]] <- pmax(pmin(df[[col_name]], upper_bound), lower_bound)
      bounds_info <- rbind(bounds_info, data.frame(Column_Name = col_name, Lower_Bound = lower_bound, Upper_Bound = upper_bound))
    }
  }
  write.csv(bounds_info, "bounds_info.csv", row.names = FALSE)
  return(df)
}

# Function to convert categorical variables to numeric using one-hot encoding or as factors
convert_categoricals <- function(df, method = "onehot") {
  if (!method %in% c("onehot", "factor")) {
    stop("Method must be either 'onehot' or 'factor'.")
  }
  for (col_name in names(df)) {
    if (is.character(df[[col_name]]) || is.factor(df[[col_name]])) {
      if (method == "factor") {
        df[[col_name]] <- as.factor(df[[col_name]])
      } else if (method == "onehot") {
        one_hot <- model.matrix(~ . - 1, data = df[col_name])
        df[[col_name]] <- NULL
        df <- cbind(df, one_hot)
      }
    }
  }
  return(df)
}

# Function to split the data into training and test sets
train_test_split <- function(df, prop = 0.75) {
  training_size <- floor(nrow(df) * prop)
  training_indices <- sample(seq_len(nrow(df)), training_size)
  training_set <- df[training_indices, ]
  test_set <- df[-training_indices, ]
  return(list(train = training_set, test = test_set))
}

