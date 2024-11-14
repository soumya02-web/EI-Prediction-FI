
---

# Prediction of EI Based on CL, WM, and RT

This project uses a machine learning model to predict EI (Emotional Intelligence) based on three features: CL (Cognitive Load), WM (Working Memory), and RT (Response Time). The repository includes the code for training the model as well as functions for making predictions on new data.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Results](#results)

## Project Overview
The goal of this project is to predict the EI value, a theoretical measure in the dataset, from the values of CL, WM, and RT. This prediction model uses polynomial transformations and is trained using nonlinear least squares regression.

## Dataset
The dataset, `data_500_eqavg.csv`, consists of 500 records with the following columns:
- `cl` - Cognitive Load
- `rt` - Response Time
- `wm` - Working Memory
- `eq` - Emotional Intelligence (target variable)

### Example Rows
| cl       | rt       | wm       | eq      |
|----------|----------|----------|---------|
| 394.55   | 340.60   | 1.52     | 39.22   |
| 386.34   | 278.84   | 1.98     | 39.20   |
| 437.21   | 520.28   | 1.43     | 39.45   |

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/soumya02-web/EI-Prediction-FI.git
   ```
2. Navigate to the project directory and install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Training the Model**: Load the notebook file `NLLSR_3_1.ipynb` and run the cells to train the model.
2. **Predicting New Data**:
   - Use the `predict_eq()` function to predict EI values for new data:
     ```python
     predict_eq("new_dataset.csv")
     ```
   - The predictions will be returned in their original scale after transformation.

3. **Evaluating Model Performance**:
   If the new data includes actual `eq` values, the function calculates the Mean Absolute Error (MAE) to evaluate the accuracy.

## Model Details
The model uses polynomial transformation on the input features and is trained using nonlinear least squares regression (NLLSR). After training, predictions can be made on scaled or transformed data.

## Results
The model performance, including metrics like Mean Absolute Error (MAE), is shown in the notebook after training. This provides an indication of the accuracy when predicting the `eq` values based on `cl`, `rt`, and `wm`.

---


