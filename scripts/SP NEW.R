# Load required libraries
library(tidyverse)
library(caret)
library(randomForest)
library(doParallel)
library(ranger)
library(e1071)
library(neuralnet)
library(xgboost)
library(lightgbm)
library(ggplot2)
library(gridExtra)
library(viridis)
library(ggpubr)
library(reshape2)
library(grid)
library(RColorBrewer)

# 1. Data Preprocessing
data <- read.csv("SolarPrediction.csv")

preprocess_data <- function(data) {
  if ("Data" %in% colnames(data) & "Time" %in% colnames(data)) {
    data$DateTime <- as.POSIXct(paste(data$Data, data$Time), format="%m/%d/%Y %H:%M:%S")
    data$Hour <- lubridate::hour(data$DateTime)
    data$Month <- lubridate::month(data$DateTime)
    data$DayOfWeek <- lubridate::wday(data$DateTime)
    
    if ("TimeSunRise" %in% colnames(data) & "TimeSunSet" %in% colnames(data)) {
      data$TimeSunRise <- as.POSIXct(data$TimeSunRise, format="%H:%M:%S")
      data$TimeSunSet <- as.POSIXct(data$TimeSunSet, format="%H:%M:%S")
      data$DaylightDuration <- as.numeric(difftime(data$TimeSunSet, data$TimeSunRise, units="hours"))
    }
    
    data <- data %>% select(-c(Data, Time, TimeSunRise, TimeSunSet, DateTime))
  } else {
    warning("The dataset does not contain 'Data' and 'Time' columns.")
  }
  
  data <- na.omit(data)
  return(data)
}

data_processed <- preprocess_data(data)

# 2. Data Visualization
ggplot(data_processed, aes(x = Radiation)) +
  geom_histogram(bins = 30, fill = "#0072B2", color = "white", alpha = 0.7) +
  labs(title = "Distribution of Solar Radiation",
       x = "Radiation (W/m²)", 
       y = "Count") +
  theme_minimal(base_size = 15) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 18),
        axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14),
        panel.grid.major = element_line(color = "grey90"),
        panel.grid.minor = element_blank())

# 3. Model Training and Evaluation with Cross-Validation
set.seed(123)
control <- trainControl(method = "cv", number = 5)

# Split the data into training and testing sets
trainIndex <- createDataPartition(data_processed$Radiation, p = .7, 
                                  list = FALSE, 
                                  times = 1)
trainSet <- data_processed[trainIndex,]
testSet <- data_processed[-trainIndex,]

# Function to calculate metrics
calculate_metrics <- function(actual, predicted) {
  rmse <- sqrt(mean((actual - predicted)^2))
  r_squared <- cor(actual, predicted)^2
  n <- length(actual)
  p <- length(coef(lm(predicted ~ actual))) - 1
  adjusted_r_squared <- 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
  mae <- mean(abs(actual - predicted))
  return(list(RMSE = rmse, R_squared = r_squared, Adjusted_R_squared = adjusted_r_squared, MAE = mae))
}

# 3.1 Random Forest
# Set up parallel processing
numCores <- detectCores() - 1  # Use one less than the total number of cores
cl <- makeCluster(numCores)
registerDoParallel(cl)

# Train Random Forest model
rf_model <- train(Radiation ~ ., data = trainSet, method = "rf", trControl = control, tuneGrid = expand.grid(mtry = c(2, 3)))

# Stop the cluster after training
stopCluster(cl)

# Make predictions on the test set
rf_predictions <- predict(rf_model, testSet)

# Calculate metrics
rf_metrics <- calculate_metrics(testSet$Radiation, rf_predictions)
cat("Random Forest Metrics:\n")
print(rf_metrics)

# 3.2 Support Vector Regression (SVR)

# Check for constant variables
constant_vars <- sapply(trainSet, function(x) length(unique(x)) == 1)
if (any(constant_vars)) {
  # Remove constant variables
  trainSet <- trainSet[, !constant_vars]
}

# Check for missing values
if (any(sapply(trainSet, function(x) any(is.na(x))))) {
  # Handle missing values
  trainSet <- trainSet[complete.cases(trainSet), ]
}

# Verify parallel computing setup
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

# Set up train control
control <- trainControl(method = "cv", number = 5, allowParallel = TRUE)

# Train SVR model
svr_model <- train(Radiation ~ ., data = trainSet, method = "svmRadial", trControl = control)

# Stop the cluster
stopCluster(cl)

# Make predictions on the test set
svr_predictions <- predict(svr_model, testSet)

# Calculate metrics
svr_metrics <- calculate_metrics(testSet$Radiation, svr_predictions)
cat("SVR Metrics:\n")
print(svr_metrics)

# 3.3 Artificial Neural Network (ANN)
# Function to calculate metrics
calculate_metrics <- function(actual, predicted) {
  if (length(actual) == 0 || length(predicted) == 0 || sd(predicted, na.rm = TRUE) == 0) {
    return(list(RMSE = NA, R_squared = NA, Adjusted_R_squared = NA, MAE = NA))
  } else {
    rmse <- sqrt(mean((actual - predicted)^2, na.rm = TRUE))
    r_squared <- cor(actual, predicted, use = "complete.obs")^2
    n <- length(actual)
    p <- length(coef(lm(predicted ~ actual))) - 1
    adjusted_r_squared <- 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
    mae <- mean(abs(actual - predicted), na.rm = TRUE)
    return(list(RMSE = rmse, R_squared = r_squared, Adjusted_R_squared = adjusted_r_squared, MAE = mae))
  }
}

# Normalize the data (min-max scaling)
normalize <- function(x) {
  if (length(na.omit(x)) == 0) {
    return(rep(NA, length(x)))  # Return NA if all values are NA
  }
  if (sd(x, na.rm = TRUE) == 0) {
    return(rep(NA, length(x)))  # Return NA if the column is constant
  }
  return ((x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE)))
}

# Check for NA values in the original test set
if (any(is.na(testSet))) {
  stop("Original test set contains NA values. Please check the data.")
}

# Normalize the training set
trainSet_normalized <- trainSet  # Start with the original training set

for (colname in colnames(trainSet)) {
  if (is.numeric(trainSet[[colname]])) {
    trainSet_normalized[[colname]] <- normalize(trainSet[[colname]])
  }
}

# Check for constant columns in the training set
constant_columns <- sapply(trainSet, function(col) {
  if (is.numeric(col) && sd(col, na.rm = TRUE) == 0) {
    return(TRUE)
  }
  return(FALSE)
})

if (any(constant_columns)) {
  cat("The following columns in the training set are constant and will lead to NA values in normalization:\n")
  print(names(trainSet)[constant_columns])
}

# Normalize the test set using the same min and max from the training set
testSet_normalized <- testSet  # Start with the original test set

for (colname in colnames(testSet)) {
  if (is.numeric(testSet[[colname]])) {
    if (colname %in% colnames(trainSet)) {
      # Normalize using training set min and max
      min_val <- min(trainSet[[colname]], na.rm = TRUE)
      max_val <- max(trainSet[[colname]], na.rm = TRUE)
      testSet_normalized[[colname]] <- (testSet[[colname]] - min_val) / (max_val - min_val)
      
      # Debugging output: Print normalized column values
      cat(sprintf("Normalized column '%s': Min = %f, Max = %f\n", colname, min(testSet_normalized[[colname]], na.rm = TRUE), max(testSet_normalized[[colname]], na.rm = TRUE)))
    } else {
      testSet_normalized[[colname]] <- rep(NA, length(testSet[[colname]]))  # Return NA if the column is not found
    }
  }
}

# Remove the Hour column from the normalized test set
testSet_normalized <- testSet_normalized[, !colnames(testSet_normalized) %in% "Hour"]

# Set the column names explicitly to ensure they are correct
colnames(testSet_normalized) <- colnames(testSet)[!colnames(testSet) %in% "Hour"]

# Check for NA values in the training set
if (any(is.na(trainSet_normalized))) {
  trainSet_normalized <- na.omit(trainSet_normalized)  # Remove rows with NA values
}

# Check for NA values in the test set
if (any(is.na(testSet_normalized))) {
  cat("Test set contains NA values. Please check the normalization process.\n")
  print(testSet_normalized[is.na(testSet_normalized)])  # Print rows with NA values
  stop("Test set contains NA values. Please check the normalization process.")
}


# Train ANN model with increased stepmax and a simpler architecture
set.seed(123)  # For reproducibility
ann_model <- neuralnet(Radiation ~ ., data = trainSet_normalized, hidden = c(3), linear.output = TRUE, threshold = 0.01, act.fct = "tanh", stepmax = 1e6)

# Check the model summary
summary(ann_model)

# Make predictions on the normalized test set
ann_predictions <- compute(ann_model, testSet_normalized)$net.result

# Check the predictions
print(head(ann_predictions))

# Ensure you have the actual values for Radiation in the test set
actual_values <- testSet$Radiation[!is.na(testSet$Radiation)]  # Ensure you have the actual values
print(head(actual_values))

# Check if lengths match
if (length(actual_values) != length(ann_predictions)) {
  stop("The length of actual values and predictions do not match!")
}

# Calculate metrics for ANN
ann_metrics <- calculate_metrics(actual_values, ann_predictions)
print(ann_metrics)


# 3.4 XGBoost
# Assuming trainSet and testSet are already defined and preprocessed

# Check column names in both datasets
cat("Training Set Columns:\n")
print(colnames(trainSet %>% select(-Radiation)))

cat("Test Set Columns:\n")
print(colnames(testSet %>% select(-Radiation)))

# Remove the 'Hour' column from the test set if it exists
testSet <- testSet %>% select(-Hour)

# Ensure that the columns are the same
if (!all(colnames(trainSet %>% select(-Radiation)) == colnames(testSet %>% select(-Radiation)))) {
  stop("Column names in training and test sets do not match!")
}

# Prepare the data for XGBoost
dtrain <- xgb.DMatrix(data = as.matrix(trainSet %>% select(-Radiation)), label = trainSet$Radiation)
dtest <- xgb.DMatrix(data = as.matrix(testSet %>% select(-Radiation)), label = testSet$Radiation)

# Set parameters for XGBoost
xgb_params <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse",
  max_depth = 6,
  eta = 0.1,
  nthread = 2
)

# Train the XGBoost model
xgb_model <- xgb.train(params = xgb_params, 
                       data = dtrain, 
                       nrounds = 100)

# Make predictions on the test set
xgb_predictions <- predict(xgb_model, dtest)

# Calculate metrics for XGBoost
calculate_metrics <- function(actual, predicted) {
  rmse <- sqrt(mean((actual - predicted)^2))
  r_squared <- cor(actual, predicted)^2
  n <- length(actual)
  p <- length(coef(lm(predicted ~ actual))) - 1
  adjusted_r_squared <- 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
  mae <- mean(abs(actual - predicted))
  return(list(RMSE = rmse, R_squared = r_squared, Adjusted_R_squared = adjusted_r_squared, MAE = mae))
}

# Calculate metrics for XGBoost
xgb_metrics <- calculate_metrics(testSet$Radiation, xgb_predictions)
cat("XGBoost Metrics:\n")
print(xgb_metrics)

# Model Interpretation and Feature Importance Analysis
importance_matrix <- xgb.importance(feature_names = colnames(trainSet %>% select(-Radiation)), model = xgb_model)
# Saving the Model for Deployment
saveRDS(xgb_model, "solar_radiation_model.rds")

# Prediction function for deployment
predict_solar_radiation <- function(new_data) {
  new_data_processed <- new_data %>% select(-Radiation, -Hour)
  new_dmatrix <- xgb.DMatrix(data = as.matrix(new_data_processed))
  prediction <- predict(xgb_model, new_dmatrix)
  return(prediction)
}

# 3.5 LightGBM
lgb_train <- lgb.Dataset(data = as.matrix(trainSet %>% select(-Radiation)), label = trainSet$Radiation)
lgb_params <- list(objective = "regression", metric = "rmse")
lgb_model <- lgb.train(params = lgb_params, data = lgb_train, nrounds = 100)
lgb_predictions <- predict(lgb_model, as.matrix(trainSet %>% select(-Radiation)))
lgb_metrics <- calculate_metrics(trainSet$Radiation, lgb_predictions)
cat("LightGBM Metrics:\n")
print(lgb_metrics)

# 4. Model Comparison
# Create a list to hold the metrics
metrics_list <- list(
  Random_Forest = rf_metrics,
  SVR = svr_metrics,
  ANN = ann_metrics,
  XGBoost = xgb_metrics,
  LightGBM = lgb_metrics
)

# Initialize an empty data frame for model comparison
model_comparison <- data.frame(Model = character(), RMSE = numeric(), R_squared = numeric(), Adjusted_R_squared = numeric(), MAE = numeric(), stringsAsFactors = FALSE)

# Loop through each model and add metrics to the comparison data frame
for (model_name in names(metrics_list)) {
  metrics <- metrics_list[[model_name]]
  
  # Check if all required metrics are present
  if (!is.null(metrics$RMSE) && !is.null(metrics$R_squared) && !is.null(metrics$Adjusted_R_squared) && !is.null(metrics$MAE)) {
    model_comparison <- rbind(model_comparison, data.frame(
      Model = model_name,
      RMSE = metrics$RMSE,
      R_squared = metrics$R_squared,
      Adjusted_R_squared = metrics$Adjusted_R_squared,
      MAE = metrics$MAE,
      stringsAsFactors = FALSE
    ))
  } else {
    cat(paste("Metrics missing for model:", model_name, "\n"))
    model_comparison <- rbind(model_comparison, data.frame(
      Model = model_name,
      RMSE = NA,
      R_squared = NA,
      Adjusted_R_squared = NA,
      MAE = NA,
      stringsAsFactors = FALSE
    ))
  }
}

# Print the model comparison data frame
print(model_comparison)

# Define a color palette
colors <- brewer.pal(n = length(unique(model_comparison$Model)), name = "Set3")

# RMSE Plot
RMSE_Plot <- ggplot(model_comparison, aes(x = reorder(Model, RMSE, na.rm = TRUE), y = RMSE, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7) +
  scale_fill_manual(values = colors) +
  labs(title = "Model Comparison: RMSE", x = "Model", y = "RMSE", fill = "Model") +
  theme_light() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
        axis.title = element_text(size = 12),
        plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
        legend.position = "bottom",
        panel.grid.major = element_line(color = "grey90"),
        panel.grid.minor = element_blank())

# R-squared Plot
R_squared_Plot <- ggplot(model_comparison, aes(x = reorder(Model, R_squared), y = R_squared, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7) +
  scale_fill_manual(values = colors) +
  labs(title = "Model Comparison: R-squared", x = "Model", y = "R-squared", fill = "Model") +
  theme_light() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
        axis.title = element_text(size = 12),
        plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
        legend.position = "bottom",
        panel.grid.major = element_line(color = "grey90"),
        panel.grid.minor = element_blank())

# Adjusted R-squared Plot
Adjusted_R_squared_Plot <- ggplot(model_comparison, aes(x = reorder(Model, Adjusted_R_squared), y = Adjusted_R_squared, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7) +
  scale_fill_manual(values = colors) +
  labs(title = "Model Comparison: Adjusted R-squared", x = "Model", y = "Adjusted R-squared", fill = "Model") +
  theme_light() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
        axis.title = element_text(size = 12),
        plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
        legend.position = "bottom",
        panel.grid.major = element_line(color = "grey90"),
        panel.grid.minor = element_blank())

# MAE Plot
MAE_Plot <- ggplot(model_comparison, aes(x = reorder(Model, MAE), y = MAE, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7) +
  scale_fill_manual(values = colors) +
  labs(title = "Model Comparison: MAE", x = "Model", y = "MAE", fill = "Model") +
  theme_light() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
        axis.title = element_text(size = 12),
        plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
        legend.position = "bottom",
        panel.grid.major = element_line(color = "grey90"),
        panel.grid.minor = element_blank())

# 5. Fine-tuning the Best Model (XGBoost assumed as best)
# Define the control for training with parallel processing
control <- trainControl(method = "cv", number = 5, allowParallel = TRUE)

# Create a tuning grid with all required parameters
tune_grid <- expand.grid(
  nrounds = c(100, 200, 300),
  max_depth = c(3, 6, 9),
  eta = c(0.01, 0.05, 0.1),
  gamma = 0,  # You can adjust this based on your needs
  colsample_bytree = 1,  # You can adjust this based on your needs
  min_child_weight = 1,  # You can adjust this based on your needs
  subsample = 1  # You can adjust this based on your needs
)

# Set up parallel processing for XGBoost tuning
cl <- makeCluster(numCores)
registerDoParallel(cl)

# Train the XGBoost model with tuning
xgb_tune <- train(Radiation ~ ., data = trainSet, method = "xgbTree", trControl = control, tuneGrid = tune_grid)

# Stop the cluster after tuning
stopCluster(cl)

# Print the best tuning parameters
print(xgb_tune$bestTune)
# 6. Training the Best Model on Combined Data
# Combine the training data (ensure you are combining correctly)
combined_data <- rbind(trainSet, trainSet)  # Make sure this is correct; usually, you would combine train and validation sets if applicable

# Train the final model using the best tuning parameters
xgb_final <- xgboost(data = as.matrix(combined_data %>% select(-Radiation)), 
                     label = combined_data$Radiation, 
                     params = as.list(xgb_tune$bestTune), 
                     nrounds = xgb_tune$bestTune$nrounds)

# 7. Final Evaluation on Test Set
# Make predictions on the test set
test_predictions <- predict(xgb_final, as.matrix(testSet %>% select(-Radiation)))

# Calculate RMSE for the test set predictions
final_rmse <- RMSE(testSet$Radiation, test_predictions)

# Function to calculate additional metrics
calculate_additional_metrics <- function(actual, predicted) {
  r_squared <- cor(actual, predicted)^2
  n <- length(actual)
  p <- length(coef(lm(predicted ~ actual))) - 1
  adjusted_r_squared <- 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
  mae <- mean(abs(actual - predicted))
  
  return(list(R_squared = r_squared, Adjusted_R_squared = adjusted_r_squared, MAE = mae))
}

# Calculate additional metrics
additional_metrics <- calculate_additional_metrics(testSet$Radiation, test_predictions)

# Print all metrics
cat("Final Evaluation Metrics:\n")
cat("RMSE:", round(final_rmse, 4), "\n")
cat("R-squared:", round(additional_metrics$R_squared, 4), "\n")
cat("Adjusted R-squared:", round(additional_metrics$Adjusted_R_squared, 4), "\n")
cat("MAE:", round(additional_metrics$MAE, 4), "\n")

# 8. Model Interpretation and Feature Importance Analysis
importance_matrix <- xgb.importance(feature_names = colnames(trainSet %>% select(-Radiation)), model = xgb_final)

ggplot(importance_matrix, aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_bar(stat = "identity", fill = "#D55E00") +
  coord_flip() +
  labs(title = "Feature Importance in XGBoost Model",
       x = "Features", 
       y = "Gain") +
  theme_minimal(base_size = 15) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 18),
        axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14),
        panel.grid.major = element_line(color = "grey90"),
        panel.grid.minor = element_blank())

# 9. Saving the Model for Deployment
saveRDS(xgb_final, "solar_radiation_model.rds")

# Prediction function for deployment
predict_solar_radiation <- function(new_data) {
  new_data_processed <- preprocess_data(new_data)
  prediction <- predict(xgb_final, as.matrix(new_data_processed %>% select(-Radiation)), iteration_range = c(0, xgb_tune$bestTune$nrounds))
  return(prediction)
}

# 10. Summary Report
summary_report <- function() {
  cat("Solar Radiation Prediction Project Summary\n")
  cat("==========================================\n\n")
  
  cat("1. Data Overview:\n")
  cat("   - Total samples:", nrow(data_processed), "\n")
  cat("   - Features:", ncol(data_processed) - 1, "\n\n")
  
  cat("2. Model Comparison:\n")
  print(model_comparison)
  cat("\n")
  
  cat("3. Best Model Performance:\n")
  cat("   - Model: XGBoost (fine-tuned)\n")
  cat("   - Final Test RMSE:", round(final_rmse, 4), "\n\n")
  
  cat("4. Top 5 Important Features:\n")
  print(head(importance_matrix, 5))
  
  cat("\n5. Model Deployment:\n")
  cat("   - Model saved as 'solar_radiation_model.rds'\n")
  cat("   - Use the 'predict_solar_radiation' function for making predictions on new data.\n")
  
  cat("\n6. Plots:\n")
  cat("   - Distribution of Solar Radiation\n")
  cat("   - Correlation Heatmap of Features\n")
  cat("   - Model Comparison: RMSE\n")
  cat("   - Feature Importance in XGBoost Model\n")
  cat("   - Predicted vs Actual Solar Radiation\n")
  cat("   - Actual vs Predicted Solar Radiation Over Time\n")
}

summary_report()
# Identify columns with zero standard deviation
constant_columns <- sapply(data_processed, function(x) sd(x, na.rm = TRUE) == 0)

# Print the names of constant columns
if (any(constant_columns)) {
  cat("Constant columns (zero standard deviation):", names(data_processed)[constant_columns], "\n")
} else {
  cat("No constant columns found.\n")
}

# Remove constant columns from the dataset
data_processed_filtered <- data_processed %>% select(-which(constant_columns))

# Calculate the correlation matrix again
correlation_matrix <- cor(data_processed_filtered %>% select_if(is.numeric), use = "complete.obs")

# Optional: Print the correlation matrix to check its contents
print(correlation_matrix)

# 1. Distribution of Solar Radiation
# 1. Distribution of Solar Radiation with Density Curve
plot1 <- ggplot(data_processed, aes(x = Radiation)) +
  geom_histogram(aes(y = ..density..), bins = 30, fill = "#0072B2", color = "white", alpha = 0.7) +
  geom_density(color = "#D55E00", size = 1.2) +  # Add density curve with increased size
  labs(title = "Distribution of Solar Radiation", x = "Radiation (W/m²)", y = "Density") +
  theme_light(base_size = 15) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 18),
        axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14),
        panel.grid.major = element_line(color = "grey90"),
        panel.grid.minor = element_blank(),
        plot.margin = ggplot2::margin(t = 10, r = 10, b = 10, l = 10))

# 2. Correlation Heatmap
plot2 <- ggplot(data = melt(correlation_matrix), aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_viridis_c(option = "C", name = "Correlation") +  # Added legend title
  labs(title = "Correlation Heatmap", x = "", y = "") +
  theme_light(base_size = 15) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
        plot.title = element_text(hjust = 0.5, face = "bold", size = 18),
        axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        plot.margin = ggplot2::margin(t = 10, r = 10, b = 10, l = 10))

# 3. Feature Importance in XGBoost Model
plot3 <- ggplot(importance_matrix, aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_bar(stat = "identity", fill = "#0072B2", width = 0.7) +  # Adjusted bar width
  coord_flip() +
  labs(title = "Feature Importance in XGBoost Model", x = "Features", y = "Gain") +
  theme_light(base_size = 15) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 18),
        axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14),
        panel.grid.major = element_line(color = "grey90"),
        panel.grid.minor = element_blank(),
        plot.margin = ggplot2::margin(t = 10, r = 10, b = 10, l = 10))

# 4. Predicted vs Actual Radiation
plot4 <- ggplot(data.frame(Actual = testSet$Radiation, Predicted = test_predictions), aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.5, color = "#0072B2") +
  geom_abline(intercept = 0, slope = 1, color = "#D55E00", linetype = "dashed", size = 1) +  # Increased line size
  labs(title = "Predicted vs Actual Radiation", x = "Actual Radiation", y = "Predicted Radiation") +
  theme_light(base_size = 15) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 18),
        axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14),
        panel.grid.major = element_line(color = "grey90"),
        panel.grid.minor = element_blank(),
        plot.margin = ggplot2::margin(t = 10, r = 10, b = 10, l = 10))

# 5. Actual vs Predicted Radiation Over Time
plot5 <- ggplot(testSet, aes(x = 1:nrow(testSet))) +
  geom_line(aes(y = Radiation, color = "Actual"), linewidth = 1.2) +  # Increased line width
  geom_line(aes(y = test_predictions, color = "Predicted"), linewidth = 1.2) +  # Increased line width
  labs(title = "Actual vs Predicted Radiation Over Time", x = "Time", y = "Radiation", color = "Legend") +
  scale_color_manual(values = c("Actual" = "#0072B2", "Predicted" = "#D55E00")) +  # Custom colors for lines
  theme_light(base_size = 15) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 18),
        axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14),
        panel.grid.major = element_line(color = "grey90"),
        panel.grid.minor = element_blank(),
        plot.margin = ggplot2::margin(t = 10, r = 10, b = 10, l = 10))

# Print the plots
print(plot1)
print(plot2)
print(plot3)
print(plot4)
print(plot5)
print(RMSE_Plot)
print(R_squared_Plot)
print(Adjusted_R_squared_Plot)
print(MAE_Plot)
