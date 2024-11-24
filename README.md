# Solar Radiation Prediction Project

## Overview
This project implements various machine learning models to predict solar radiation based on meteorological data. The project uses advanced algorithms such as Random Forest, Support Vector Regression (SVR), Artificial Neural Networks (ANN), XGBoost, and LightGBM to provide accurate solar radiation predictions.

---

## Table of Contents
- Features
- Requirements
- Installation
- Usage
- Model Performance
- Project Structure
- Data Preprocessing
- Models Implemented
- Results Visualization
- Contributing
- License

---

## Features
- Comprehensive data preprocessing pipeline.
- Implementation of multiple machine learning models.
- Cross-validation for robust model evaluation.
- Feature importance analysis.
- Detailed visualization of results.
- Model comparison metrics.
- Parallel processing support for faster training.
- Production-ready prediction function.

---

## Requirements
The following R libraries are required:

```r
# Core Libraries
tidyverse
caret
doParallel
ggplot2
gridExtra
viridis
ggpubr
reshape2
grid
RColorBrewer

# Machine Learning Libraries
randomForest
ranger
e1071
neuralnet
xgboost
lightgbm
```

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/OmkeshLamb2004/SolarRadiation_Prediction
   cd solar-radiation-prediction
   ```

2. **Install required R packages:**

   ```r
   install.packages(c("tidyverse", "caret", "randomForest", "doParallel", 
                      "ranger", "e1071", "neuralnet", "xgboost", "lightgbm", 
                      "ggplot2", "gridExtra", "viridis", "ggpubr", "reshape2", 
                      "grid", "RColorBrewer"))
   ```

---

## Usage

1. Prepare your data in CSV format with the following columns:
   - Date
   - Time
   - Radiation
   - Temperature
   - Pressure
   - Humidity
   - WindDirection (in Degrees)
   - Wind Speed
   - TimeSunRise
   - TimeSunSet

2. Run the main script:

   ```r
   source("solar_radiation_prediction.R")
   ```

3. For predictions on new data:

   ```r
   new_data <- read.csv("new_data.csv")
   predictions <- predict_solar_radiation(new_data)
   ```

---

## Model Performance

The project implements and compares the following models:
- **Random Forest**
- **Support Vector Regression (SVR)**
- **Artificial Neural Network (ANN)**
- **XGBoost**
- **LightGBM**

### Performance Metrics:
- RMSE (Root Mean Square Error)
- R-squared
- Adjusted R-squared
- MAE (Mean Absolute Error)

---

## Project Structure

```plaintext
solar-radiation-prediction/
├── data/
│   └── SolarPrediction.csv
├── scripts/
│   ├── solar_radiation_prediction.R
│   
├── models/
│   └── solar_radiation_model.rds
├── results/
│   └── plots/
├── README.md
└── LICENSE
```

---

## Data Preprocessing

The preprocessing pipeline includes:
- Handling missing values.
- Feature engineering (e.g., time-based features).
- Data normalization.
- Outlier detection.
- Calculation of daylight duration.

---

## Models Implemented

1. **Random Forest**
   - Handles non-linear relationships.
   - Provides feature importance ranking.

2. **Support Vector Regression (SVR)**
   - Uses radial basis function kernel.
   - Includes cross-validated hyperparameter tuning.

3. **Artificial Neural Network (ANN)**
   - Multi-layer perceptron architecture.
   - Normalized input features.

4. **XGBoost**
   - Gradient boosting implementation.
   - Advanced feature importance analysis.

5. **LightGBM**
   - Light Gradient Boosting Machine.
   - Efficient handling of large datasets.

---

## Results Visualization

The project includes various visualizations:
- Distribution of solar radiation.
- Correlation heatmap.
- Feature importance plots.
- Model comparison charts.
- Actual vs Predicted plots.
- Time series analysis plots.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

