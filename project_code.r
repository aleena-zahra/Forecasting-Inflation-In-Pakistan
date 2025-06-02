# Load required libraries for data manipulation, visualization, and modeling
library(readr)         # For reading CSV files
library(dplyr)         # For data wrangling
library(zoo)           # For time series functions like na.locf
library(ggplot2)       # For creating plots
library(forecast)      # For ARIMA modeling
library(glmnet)        # For Ridge, Lasso, and Elastic Net regression
library(GGally)        # For ggpairs plot (scatterplot matrix)
library(caret)         # For model training utilities
library(e1071)         # For SVM
library(randomForest)  # For Random Forest
library(tseries)
library(tidyr)
# Load the training and testing datasets
train_df <- read.csv('train - Copy.csv', stringsAsFactors = FALSE)
test_df <- read.csv('test - Copy.csv', stringsAsFactors = FALSE)

# Replace missing values denoted by ".." with NA
train_df[train_df == ".."] <- NA
test_df[test_df == ".."] <- NA

# Convert all columns (except first) to numeric
num_cols <- names(train_df)[2:ncol(train_df)]
for (col in num_cols) {
  train_df[[col]] <- as.numeric(train_df[[col]])
  test_df[[col]] <- as.numeric(test_df[[col]])
}

# Select only numeric columns
numeric_df <- train_df[sapply(train_df, is.numeric)]

# Impute missing values with forward fill
train_df[num_cols] <- lapply(train_df[num_cols], na.locf, na.rm = FALSE)
test_df[num_cols] <- lapply(test_df[num_cols], na.locf, na.rm = FALSE)

# View correlation matrix
cor(train_df[sapply(train_df, is.numeric)])

# Display summary statistics
summary(train_df)

# Create a boxplot of numeric variables
boxplot(numeric_df, main = "Boxplot of Numeric Variables", ylab = "Values", col = rainbow(ncol(numeric_df)))

# Create a scatterplot matrix
ggpairs(numeric_df)

# Save target column name and its stats
target_col <- "Inflation..consumer.prices..annual.....FP.CPI.TOTL.ZG."
train_target_mean <- mean(train_df[[target_col]], na.rm = TRUE)
train_target_sd <- sd(train_df[[target_col]], na.rm = TRUE)

# Z-score scale all numeric columns using training stats
train_means <- colMeans(train_df[num_cols], na.rm = TRUE)
train_sds <- apply(train_df[num_cols], 2, sd, na.rm = TRUE)
train_df[num_cols] <- scale(train_df[num_cols])
test_df[num_cols] <- sweep(sweep(test_df[num_cols], 2, train_means, "-"), 2, train_sds, "/")

# Prepare train/test feature matrices and target vectors
X_train <- train_df[, setdiff(num_cols, target_col)]
y_train <- train_df[[target_col]]
X_test <- test_df[, setdiff(num_cols, target_col)]
y_test <- test_df[[target_col]]

# Check for missing values
cat("NAs in X_train:", sum(is.na(X_train)), "\n")
cat("NAs in X_test:", sum(is.na(X_test)), "\n")

# Impute remaining NAs with mean
X_train_ready <- makeX(X_train, na.impute = TRUE, fill.na = "mean")
X_test_ready <- makeX(X_test, na.impute = TRUE, fill.na = "mean")

# Convert data to matrices for modeling
X_train_matrix <- as.matrix(X_train_ready)
X_test_matrix <- as.matrix(X_test_ready)

# Fit models
set.seed(123)
ridge_cv <- cv.glmnet(X_train_matrix, y_train, alpha = 0)
lasso_cv <- cv.glmnet(X_train_matrix, y_train, alpha = 1)
elastic_cv <- cv.glmnet(X_train_matrix, y_train, alpha = 0.5)
svm_fit <- svm(y_train ~ ., data = as.data.frame(X_train_matrix), kernel = "radial", cost = 1, gamma = 0.1)
rf_fit <- randomForest(y_train ~ ., data = as.data.frame(X_train_matrix), ntree = 100, na.action = na.omit)

# Predict using models
ridge_preds <- predict(ridge_cv, s = "lambda.min", newx = X_test_matrix)
lasso_preds <- predict(lasso_cv, s = "lambda.min", newx = X_test_matrix)
elastic_preds <- predict(elastic_cv, s = "lambda.min", newx = X_test_matrix)
svm_preds <- predict(svm_fit, as.data.frame(X_test_matrix))
rf_preds <- predict(rf_fit, as.data.frame(X_test_matrix))

# Actual test values (original scale)
real_test_values <- c(5.078,10.578,9.740, 9.496, 19.874, 30.768, 12.633)

# Scale actual test values using training stats
scaled_actuals <- (real_test_values - train_target_mean) / train_target_sd

# Fit ARIMA model on inflation data
# Check for stationarity using Augmented Dickey-Fuller (ADF) test
target_col <- "Inflation..consumer.prices..annual.....FP.CPI.TOTL.ZG."
target_ts <- ts(train_df[[target_col]], frequency = 1)

adf_test_result <- adf.test(target_ts)
cat("ADF Test p-value:", adf_test_result$p.value, "\n")

# If p-value > 0.05, the series is non-stationary, so apply differencing
if (adf_test_result$p.value > 0.05) {
  cat("The series is non-stationary. Differencing will be applied.\n")
  
  # Apply first-order differencing
  target_ts_diff1 <- diff(target_ts, differences = 1)
  
  # Check stationarity after first differencing
  adf_test_diff1_result <- adf.test(target_ts_diff1)
  cat("ADF Test p-value after first differencing:", adf_test_diff1_result$p.value, "\n")
  
  # If still non-stationary (p-value > 0.05), apply second-order differencing
  if (adf_test_diff1_result$p.value > 0.05) {
    cat("The series is still non-stationary. Second differencing will be applied.\n")
    target_ts_diff2 <- diff(target_ts_diff1, differences = 1)
    
    # Check stationarity after second differencing
    adf_test_diff2_result <- adf.test(target_ts_diff2)
    cat("ADF Test p-value after second differencing:", adf_test_diff2_result$p.value, "\n")
    
    # If still non-stationary (p-value > 0.05), apply third-order differencing
    if (adf_test_diff2_result$p.value > 0.05) {
      cat("The series is still non-stationary. Third differencing will be applied.\n")
      target_ts_diff3 <- diff(target_ts_diff2, differences = 1)
      
      # Check stationarity after third differencing
      adf_test_diff3_result <- adf.test(target_ts_diff3)
      cat("ADF Test p-value after third differencing:", adf_test_diff3_result$p.value, "\n")
      
      target_ts_diff3 <- target_ts_diff3  # Use third differencing if still non-stationary
    } else {
      cat("The series is stationary after second differencing.\n")
      target_ts_diff3 <- target_ts_diff2  # No need for third differencing
    }
  } else {
    cat("The series is stationary after first differencing.\n")
    target_ts_diff3 <- target_ts_diff2  # No need for second or third differencing
  }
} else {
  cat("The series is already stationary.\n")
  target_ts_diff3 <- target_ts  # No differencing needed
}

# Fit the ARIMA model on the differenced series
arima_fit <- auto.arima(target_ts_diff2)

# Forecast for the next 7 years (matching the test set)
arima_forecast <- forecast(arima_fit, h = 7)
arima_preds <- arima_forecast$mean


# Combine predictions into a dataframe
predictions <- data.frame(
  Year = 2018:2024,
  Ridge = as.numeric(ridge_preds),
  Lasso = as.numeric(lasso_preds),
  ElasticNet = as.numeric(elastic_preds),
  ARIMA = as.numeric(arima_preds),
  Actual_Scaled = scaled_actuals,
  SVM = as.numeric(svm_preds),
  RandomForest = as.numeric(rf_preds)
)

# Print predictions
print(predictions)

# Calculate RMSE for each model
cat("\nModel Performance (RMSE in scaled space):\n")
cat("Ridge:", sqrt(mean((predictions$Ridge - predictions$Actual_Scaled)^2)), "\n")
cat("Lasso:", sqrt(mean((predictions$Lasso - predictions$Actual_Scaled)^2)), "\n")
cat("ElasticNet:", sqrt(mean((predictions$ElasticNet - predictions$Actual_Scaled)^2)), "\n")
cat("ARIMA:", sqrt(mean((predictions$ARIMA - predictions$Actual_Scaled)^2)), "\n")
cat("SVM:", sqrt(mean((predictions$SVM - predictions$Actual_Scaled)^2)), "\n")
cat("RandomForest:", sqrt(mean((predictions$RandomForest - predictions$Actual_Scaled)^2)), "\n")

# Inverse transform predictions to original scale
predictions$Ridge_Original <- (predictions$Ridge * train_target_sd) + train_target_mean
predictions$Lasso_Original <- (predictions$Lasso * train_target_sd) + train_target_mean
predictions$ElasticNet_Original <- (predictions$ElasticNet * train_target_sd) + train_target_mean
predictions$ARIMA_Original <- (predictions$ARIMA * train_target_sd) + train_target_mean
predictions$Actual_Original <- real_test_values
predictions$SVM_Original <- (predictions$SVM * train_target_sd) + train_target_mean
predictions$RandomForest_Original <- (predictions$RandomForest * train_target_sd) + train_target_mean

# Print predictions in original scale
print(predictions[, c("Year", "Ridge_Original", "Lasso_Original", 
                      "ElasticNet_Original", "ARIMA_Original", "Actual_Original",
                      "SVM_Original", "RandomForest_Original")])

# Plot predictions vs actual for each model
# Plot Ridge vs Actual
ggplot(predictions, aes(x = Year)) +
  geom_line(aes(y = Ridge_Original, color = "Ridge"), size = 1.2) +
  geom_line(aes(y = Actual_Original, color = "Actual"), size = 1.2)+
  labs(title = "Model Predictions vs Actual Values", x = "Year", y = "Inflation Rate (Original Scale)") +
  theme_minimal() +
  scale_color_manual(values = c("Ridge" = "blue", "Lasso" = "green", 
                                "ElasticNet" = "red", "ARIMA" = "purple", 
                                "Actual" = "orange"))


# Plot Lasso vs Actual
ggplot(predictions, aes(x = Year)) +
  geom_line(aes(y = Lasso_Original, color = "Lasso"), size = 1.2) +
  geom_line(aes(y = Actual_Original, color = "Actual"), size = 1.2)+
  labs(title = "Model Predictions vs Actual Values", x = "Year", y = "Inflation Rate (Original Scale)") +
  theme_minimal() +
  scale_color_manual(values = c("Ridge" = "blue", "Lasso" = "green", 
                                "ElasticNet" = "red", "ARIMA" = "purple", 
                                "Actual" = "orange"))


# Plot ElasticNet vs Actual
ggplot(predictions, aes(x = Year)) +
  geom_line(aes(y = ElasticNet_Original, color = "ElasticNet"), size = 1.2) +
  geom_line(aes(y = Actual_Original, color = "Actual"), size = 1.2)+
  labs(title = "Model Predictions vs Actual Values", x = "Year", y = "Inflation Rate (Original Scale)") +
  theme_minimal() +
  scale_color_manual(values = c("Ridge" = "blue", "Lasso" = "green", 
                                "ElasticNet" = "red", "ARIMA" = "purple", 
                                "Actual" = "orange"))


# Plot ARIMA vs Actual
ggplot(predictions, aes(x = Year)) +
  geom_line(aes(y = ARIMA_Original, color = "ARIMA"), size = 1.2) +
  geom_line(aes(y = Actual_Original, color = "Actual"), size = 1.2)+
  labs(title = "Model Predictions vs Actual Values", x = "Year", y = "Inflation Rate (Original Scale)") +
  theme_minimal() +
  scale_color_manual(values = c("Ridge" = "blue", "Lasso" = "green", 
                                "ElasticNet" = "red", "ARIMA" = "purple", 
                                "Actual" = "orange"))

# Plot SVM vs Actual
ggplot(predictions, aes(x = Year)) +
  geom_line(aes(y = SVM_Original, color = "SVM"), size = 1.2) +
  geom_line(aes(y = Actual_Original, color = "Actual"), size = 1.2) +
  labs(title = "SVM Predictions vs Actual Values",
       x = "Year", y = "Inflation Rate (Original Scale)") +
  theme_minimal() +
  scale_color_manual(values = c("SVM" = "blue", "Actual" = "orange"))


# Plot RandomForest vs Actual 
ggplot(predictions, aes(x = Year)) +
  geom_line(aes(y = RandomForest_Original, color = "Random Forest"), size = 1.2) +
  geom_line(aes(y = Actual_Original, color = "Actual"), size = 1.2) +
  labs(title = "Random Forest Predictions vs Actual Values",
       x = "Year", y = "Inflation Rate (Original Scale)") +
  theme_minimal() +
  scale_color_manual(values = c("Random Forest" = "darkgreen", "Actual" = "orange"))

# Combined plot
ggplot(predictions, aes(x = Year)) +
  geom_line(aes(y = Ridge_Original, color = "Ridge"), size = 1.2) +
  geom_line(aes(y = Lasso_Original, color = "Lasso"), size = 1.2) +
  geom_line(aes(y = ElasticNet_Original, color = "ElasticNet"), size = 1.2) +
  geom_line(aes(y = ARIMA_Original, color = "ARIMA"), size = 1.2) +
  geom_line(aes(y = RandomForest_Original, color = "Random Forest"), size = 1.2) +
  geom_line(aes(y = SVM_Original, color = "SVM"), size = 1.2) +
  geom_line(aes(y = Actual_Original, color = "Actual"), size = 1.2) +
  labs(title = "Model Predictions vs Actual Values",
       x = "Year", y = "Inflation Rate (Original Scale)") +
  theme_minimal() +
  scale_color_manual(values = c("Ridge" = "blue", "Lasso" = "green", 
                                "ElasticNet" = "red", "ARIMA" = "purple", 
                                "Random Forest" = "darkgreen", "SVM" = "brown", 
                                "Actual" = "orange"))


#####################################################################################
# additional graphs
# Bar plot of RMSE values
library(tibble)

rmse_df <- tibble::tibble(
  Model = c("Ridge", "Lasso", "ElasticNet", "ARIMA", "SVM", "Random Forest"),
  RMSE = c(
    sqrt(mean((predictions$Ridge - predictions$Actual_Scaled)^2)),
    sqrt(mean((predictions$Lasso - predictions$Actual_Scaled)^2)),
    sqrt(mean((predictions$ElasticNet - predictions$Actual_Scaled)^2)),
    sqrt(mean((predictions$ARIMA - predictions$Actual_Scaled)^2)),
    sqrt(mean((predictions$SVM - predictions$Actual_Scaled)^2)),
    sqrt(mean((predictions$RandomForest - predictions$Actual_Scaled)^2))
  )
)

ggplot(rmse_df, aes(x = reorder(Model, RMSE), y = RMSE, fill = Model)) +
  geom_col(width = 0.7) +
  coord_flip() +
  labs(title = "RMSE Comparison of Models", x = "Model", y = "RMSE") +
  theme_minimal() +
  scale_fill_brewer(palette = "Set1")

# Add absolute error columns
predictions$Elasticnet_Error <- abs(predictions$ElasticNet_Original - predictions$Actual_Original)
predictions$Ridge_Error <- abs(predictions$Ridge_Original - predictions$Actual_Original)
predictions$Lasso_Error <- abs(predictions$Lasso_Original - predictions$Actual_Original)
predictions$ARIMA_Error <- abs(predictions$ARIMA_Original - predictions$Actual_Original)
predictions$SVM_Error <- abs(predictions$SVM_Original - predictions$Actual_Original)
predictions$RandomForest_Error <- abs(predictions$RandomForest_Original - predictions$Actual_Original)

# Plot error bars
ggplot(predictions, aes(x = Year)) +
  geom_line(aes(y = Elasticnet_Error, color = "Elasticnet"), size = 1.2) +
  geom_line(aes(y = Ridge_Error, color = "Ridge"), size = 1.2) +
  geom_line(aes(y = Lasso_Error, color = "Lasso"), size = 1.2) +
  geom_line(aes(y = ARIMA_Error, color = "ARIMA"), size = 1.2) +
  geom_line(aes(y = SVM_Error, color = "SVM"), size = 1.2) +
  geom_line(aes(y = RandomForest_Error, color = "RandomForest"), size = 1.2) +
  labs(title = "Absolute Error by Year", y = "Absolute Error (Inflation %)", x = "Year") +
  theme_minimal() +
  scale_color_manual(values = c("Lasso" = "green", "ARIMA" = "purple" , "Ridge" = "orange", "Elasticnet" = "yellow", "SVM" = "blue", "RandomForest" = "brown"))


################################

# Calculate residuals
predictions$Ridge_Residual <- predictions$Ridge_Original - predictions$Actual_Original
predictions$Lasso_Residual <- predictions$Lasso_Original - predictions$Actual_Original
predictions$ElasticNet_Residual <- predictions$ElasticNet_Original - predictions$Actual_Original
predictions$ARIMA_Residual <- predictions$ARIMA_Original - predictions$Actual_Original
predictions$SVM_Residual <- predictions$SVM_Original - predictions$Actual_Original
predictions$RandomForest_Residual <- predictions$RandomForest_Original - predictions$Actual_Original

# Reshape to long format

residuals_long <- predictions %>%
  select(Year, Ridge_Residual, Lasso_Residual, ElasticNet_Residual, ARIMA_Residual, SVM_Residual, RandomForest_Residual) %>%
  pivot_longer(-Year, names_to = "Model", values_to = "Residual")

# Clean model names
residuals_long$Model <- gsub("_Residual", "", residuals_long$Model)

# Plot residuals
ggplot(residuals_long, aes(x = Year, y = Residual, color = Model)) +
  geom_line(size = 1.2) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(title = "Residual Plot: Predictions vs Actual Inflation",
       y = "Residual (Prediction - Actual)", x = "Year") +
  theme_minimal() +
  scale_color_brewer(palette = "Set1")

####################################################
# Coefficients from Lasso
lasso_coef <- coef(lasso_cv, s = "lambda.min")
lasso_coef_df <- data.frame(
  Feature = rownames(lasso_coef),
  Coefficient = as.vector(lasso_coef)
)
# Filter out zero coefficients and intercept
lasso_coef_df <- subset(lasso_coef_df, Coefficient != 0 & Feature != "(Intercept)")

# Plot
ggplot(lasso_coef_df, aes(x = reorder(Feature, Coefficient), y = Coefficient)) +
  geom_col(fill = "purple") +
  coord_flip() +
  labs(title = "Lasso Feature Importance", x = "Feature", y = "Coefficient") +
  theme_minimal() 
####
# Training data starts in 1985 and ends in 2017, so we have 33 years
train_start_year <- 1985
train_end_year <- 2017
train_len <- 33  # This matches your data

# Forecast for 7 test years (2018–2024)
test_years <- 2018:2024  # 7 years

# Combine years correctly
years_all <- c(train_start_year:train_end_year, test_years)  # 33 years from training + 7 years for forecast = 40 years

# Ensure actual data has 33 values from training + 7 test values
actual_all <- c(train_df[[target_col]], real_test_values)  # 33 training + 7 test = 40 actual values

# Make sure that arima_pred_all contains both fitted and forecasted values
arima_pred_all <- c(as.numeric(fitted_vals), as.numeric(future_vals))  # Combine fitted values and forecast

# Sanity check: lengths should now match
stopifnot(length(years_all) == length(arima_pred_all), length(arima_pred_all) == length(actual_all))

# Create data frame for plotting
arima_plot_df <- data.frame(
  Year = years_all,
  Actual = actual_all,
  ARIMA_Predicted = arima_pred_all
)

ggplot(arima_plot_df, aes(x = Year)) +
  geom_line(aes(y = Actual, color = "Actual"), size = 1) +  # Plot actual values
  geom_line(aes(y = ARIMA_Predicted, color = "ARIMA Predicted"), size = 1, linetype = "dashed") +  # Plot ARIMA predictions
  labs(title = "ARIMA Predictions",
       x = "Year",
       y = "Value") +
  scale_color_manual(values = c("Actual" = "blue", "ARIMA Predicted" = "red")) +  # Customize colors
  theme_minimal()  # Clean and minimal theme

