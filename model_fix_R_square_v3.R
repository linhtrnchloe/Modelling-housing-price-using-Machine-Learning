# Clear environment
rm(list=ls())

# Install and load packages
install.packages(c("sf", "reshape2", "rpart", "rpart.plot", "caret", "randomForest", "pROC", "lightgbm", "dplyr", "lubridate", "glmnet","ggplot2", "ggmap"))
library(sf)
library(reshape2)
library(rpart)
library(rpart.plot)
library(caret)
library(randomForest)
library(pROC)
library(lightgbm)
library(dplyr)
library(lubridate)
library(glmnet)
library(ggplot2)
library(ggmap)
# Load Austin housing data
df <- read.csv("C:/Den Lion/austinHousingData.csv")

# Load mortgage rate data
mortgage_df <- read.csv("C:/Den Lion/MORTGAGE30US.csv")

# Diagnostics (from previous)
cat("=== Austin Dataset Diagnostics ===\n")
cat("latest_saledate sample (raw):\n")
print(head(df$latest_saledate, 10))
cat("latest_saledate NA count (raw):", sum(is.na(df$latest_saledate)), "\n")
cat("\n=== Mortgage Dataset Diagnostics ===\n")
cat("Column names:\n")
print(names(mortgage_df))
cat("observation_date sample (raw):\n")
print(head(mortgage_df$observation_date, 10))
cat("MORTGAGE30US sample (raw):\n")
print(head(mortgage_df$MORTGAGE30US, 10))
cat("Rows before cleaning:", nrow(mortgage_df), "\n")
cat("MORTGAGE30US NA count (raw):", sum(is.na(mortgage_df$MORTGAGE30US)), "\n")

# Select relevant columns
data <- subset(df, select = c(latestPrice, propertyTaxRate, garageSpaces, lotSizeSqFt, avgSchoolDistance, avgSchoolRating, numOfBathrooms, numOfBedrooms, latitude, longitude, latest_saledate, livingAreaSqFt))

# Clean data: Remove NA and invalid coordinates
data <- data[complete.cases(data[, c("latestPrice", "latitude", "longitude", "latest_saledate", "livingAreaSqFt")]), ]
data <- data[data$latitude >= 30.0 & data$latitude <= 30.5 & data$longitude >= -98.0 & data$longitude <= -97.5 & data$latestPrice > 0, ]

# Preprocessing: Handle outliers
# Cap latestPrice at 99th percentile
price_cap <- quantile(data$latestPrice, 0.99)
data$latestPrice[data$latestPrice > price_cap] <- price_cap
cat("Capped latestPrice at 99th percentile:", price_cap, "\n")

# Feature Engineering
# Price per square foot
# data$price_per_sqft <- data$latestPrice / data$livingAreaSqFt
# Year of sale
data$latest_saledate <- as.Date(data$latest_saledate, format = "%Y-%m-%d")
data$sale_year <- as.numeric(format(data$latest_saledate, "%Y"))
# Spatial clustering (simplified neighborhood proxy)
library(sp)
coords <- as.matrix(data[, c("longitude", "latitude")])
kmeans_clusters <- kmeans(coords, centers = 10, nstart = 20)
data$neighborhood_cluster <- as.factor(kmeans_clusters$cluster)


# Merge mortgage data
mortgage_df <- mortgage_df[complete.cases(mortgage_df[, c("observation_date", "MORTGAGE30US")]), ]
mortgage_df$observation_date <- as.Date(mortgage_df$observation_date, format = "%Y-%m-%d")
if (nrow(mortgage_df) == 0) stop("Empty mortgage dataset.")
data$mortgage_rate <- sapply(data$latest_saledate, function(sale_date) {
  if (is.na(sale_date)) return(NA)
  date_diffs <- abs(mortgage_df$observation_date - sale_date)
  nearest_idx <- which.min(date_diffs)
  if (length(nearest_idx) == 0 || date_diffs[nearest_idx] > 180) return(NA)
  mortgage_df$MORTGAGE30US[nearest_idx]
})
# Impute missing mortgage rates
if (any(is.na(data$mortgage_rate))) {
  median_rate <- median(data$mortgage_rate, na.rm = TRUE)
  if (is.na(median_rate)) median_rate <- 3.85
  data$mortgage_rate[is.na(data$mortgage_rate)] <- median_rate
}

# Calculate distance to river
homes_sf <- st_as_sf(data, coords = c("longitude", "latitude"), crs = 4326, remove = FALSE)
river_coords <- data.frame(
  longitude = c(-97.80, -97.78, -97.75, -97.74, -97.70, -97.65),
  latitude = c(30.35, 30.30, 30.27, 30.26, 30.25, 30.24)
)
river_sf <- st_sf(geometry = st_sfc(st_linestring(as.matrix(river_coords[, c("longitude", "latitude")])), crs = 4326))
homes_sf <- st_transform(homes_sf, 32614)
river_sf <- st_transform(river_sf, 32614)
data$distance_to_river_km <- as.numeric(st_distance(homes_sf, river_sf)) / 1000
data$distance_to_river_km <- round(data$distance_to_river_km, 2)

# Scale numeric features
numeric_cols <- c("latestPrice", "propertyTaxRate", "garageSpaces", "lotSizeSqFt", "avgSchoolDistance", "avgSchoolRating", "numOfBathrooms", "numOfBedrooms", "distance_to_river_km", "mortgage_rate", "sale_year")
data[numeric_cols] <- scale(data[numeric_cols])

# Save updated dataset
write.csv(data, "C:/Den Lion/austinHousingData_with_features_draft.csv", row.names = FALSE)

# Prepare for modeling
data$latestPrice_1 <- data$latestPrice / 10000
data$latestPrice <- NULL
data$latitude <- NULL
data$longitude <- NULL
data$latest_saledate <- NULL
data$propertyTaxRate <- NULL
# data$livingAreaSqFt <- NULL

# Split data
set.seed(122644)
sample_index <- sample(1:nrow(data), size = 0.8 * nrow(data))
train_set <- data[sample_index, ]
validation_set <- data[-sample_index, ]

# BUILDING MODELS
# Linear Regression with L2 (Ridge)
x_train <- as.matrix(train_set[, !names(train_set) %in% "latestPrice_1"])
y_train <- train_set$latestPrice_1
ridge <- glmnet(x_train, y_train, alpha = 0, lambda = 0.1)

# Decision Tree (more pruning)
t.regr <- rpart(latestPrice_1 ~ ., data = train_set, cp = 0.02, maxdepth = 10)
rpart.plot(t.regr, under = FALSE, fallen.leaves = FALSE, cex = 0.9)

# Random Forest (more regularization)
rf <- randomForest(latestPrice_1 ~ ., data = train_set, ntree = 500, mtry = 4, nodesize = 20, maxnodes = 50)
plot(rf)
rf_simplified <- randomForest(latestPrice_1 ~ ., data = train_set, ntree = 100, mtry = 4, nodesize = 20, maxnodes = 50)
plot(rf_simplified)

# LightGBM (increased regularization)
dtrain <- lgb.Dataset(as.matrix(train_set[, !names(train_set) %in% "latestPrice_1"]), label = train_set$latestPrice_1)
dval <- lgb.Dataset(as.matrix(validation_set[, !names(validation_set) %in% "latestPrice_1"]), label = validation_set$latestPrice_1)
params <- list(objective = "regression", metric = "rmse", max_depth = 4, min_data_in_leaf = 30, learning_rate = 0.03, lambda_l1 = 1, lambda_l2 = 1)
lgb_model <- lgb.train(params, dtrain, nrounds = 200, valids = list(validation = dval), early_stopping_rounds = 20)

# Hyperparameter Tuning (expanded)
train_control <- trainControl(method = "cv", number = 5)
tune_grid <- expand.grid(mtry = c(3, 5, 7), splitrule = "variance", min.node.size = c(10, 20, 30))
rf_tuned <- train(latestPrice_1 ~ ., data = train_set, method = "ranger", trControl = train_control, tuneGrid = tune_grid, num.trees = 100)

# Evaluation function
Evaluate_Model <- function(model_list, data, target_col) {
  results <- list()
  residuals <- lapply(model_list, function(m) {
    if (inherits(m, "glmnet")) {
      data[[target_col]] - predict(m, as.matrix(data[, !names(data) %in% target_col]))
    } else if (inherits(m, "lgb.Booster")) {
      data[[target_col]] - predict(m, as.matrix(data[, !names(data) %in% target_col]))
    } else {
      data[[target_col]] - predict(m, data)
    }
  })
  results$RSS <- sapply(residuals, function(r) sum(r^2))
  results$MAE <- sapply(residuals, function(r) mean(abs(r)))
  results$RMSE <- sapply(residuals, function(r) sqrt(mean(r^2)))
  mean_target <- mean(data[[target_col]])
  results$RAE <- sapply(residuals, function(r) sum(abs(r)) / sum(abs(data[[target_col]] - mean_target)))
  results$RRSE <- sapply(residuals, function(r) sqrt(sum(r^2) / sum((data[[target_col]] - mean_target)^2)))
  results$R2 <- 1 - sapply(residuals, function(r) sum(r^2) / sum((data[[target_col]] - mean_target)^2))
  return(as.data.frame(results))
}

# Define models
model <- list(
  #"Ridge Regression" = ridge,
  "Decision Tree (cp=0.02)" = t.regr,
  "Random Forest" = rf,
  "RF simpliefied" = rf_simplified,
 # "LightGBM" = lgb_model,
  "Random Forest Tuned" = rf_tuned
)

# Evaluate
train_metrics <- Evaluate_Model(model, train_set, "latestPrice_1")
cat("=== Training Set Metrics ===\n")
print(train_metrics)
write.csv(train_metrics, "C:/Den Lion/train_metrics_2.csv", row.names = TRUE)

validation_metrics <- Evaluate_Model(model, validation_set, "latestPrice_1")
cat("\n=== Validation Set Metrics ===\n")
print(validation_metrics)
write.csv(validation_metrics, "C:/Den Lion/validation_metrics_2.csv", row.names = TRUE)
# Feature Importance (Random Forest)
cat("\n=== Random Forest Feature Importance ===\n")
print(rf$importance)

# Test model without new features
data_minimal <- subset(data, select = -c(distance_to_river_km, mortgage_rate, sale_year, neighborhood_cluster))
train_minimal <- data_minimal[sample_index, ]
validation_minimal <- data_minimal[-sample_index, ]
rf_minimal <- randomForest(latestPrice_1 ~ ., data = train_minimal, ntree = 100, mtry = 4, nodesize = 20, maxnodes = 50)
model_minimal <- list("Random Forest Minimal" = rf_minimal)
train_metrics_minimal <- Evaluate_Model(model_minimal, train_minimal, "latestPrice_1")
validation_metrics_minimal <- Evaluate_Model(model_minimal, validation_minimal, "latestPrice_1")
cat("\n=== Minimal Model Metrics ===\n")
print(rbind(Train = train_metrics_minimal, Validation = validation_metrics_minimal))