# Stock Price Prediction with LSTM and Artificial Rabbits Optimization (ARO)

This repository implements a deep LSTM network optimized with the Artificial Rabbits Optimization (ARO) algorithm to forecast the next-day closing price of the S&P 500 ETF (VOO) using historical data from 2018-01-01 to 2023-01-01. The model is evaluated across multiple window sizes (5, 10, 30, 60, 120) and includes cross-validation.

## Overview
- **Model**: LSTM with dynamically optimized architecture and hyperparameters using ARO.
- **Features**: Ticker, date range, window size, population and iteration can be adjusted in the main program. (addition to number of fold for cross validation only)

### Key Components:
- Standardization of closing price
- Time-series windowing with variable window sizes.
- ARO-based hyperparameter optimization (neurons, dropout, optimizer, learning rate).
- Performance metrics: MSE, RMSE, MAE, MAPE, R².
- Cross-validation with time-preserved folds.
- Visualization: Training/validation loss curves, prediction vs. actual scatter plots, and grouped bar charts for metrics.

## Installation

### Dependencies
- **Python**: 3.9+
- **Libraries**:
```bash
  - yfinance==0.2.54
  - tensorflow==2.10.0
  - scikit-learn==1.6.1
  - pandas==2.2.3
  - numpy==1.23.5
```

### Hardware
- CUDA-enabled GPU (e.g., NVIDIA RTX 2060) recommended for CuDNNLSTM acceleration.

## Project Structure

- **Main Script**: Single Jupyter notebook (`main.ipynb`) containing:
  - Data download (`yfinance`).
  - Train/test split (80/20 ratio).
  - Window creation and standardization (`StandardScaler`).
  - ARO optimization loop.
  - LSTM model training and evaluation.
  - Cross-validation (3 folds).
  - Visualization functions (`matplotlib`).

## Methodology

### Data Preparation
- **Input**: Historical closing prices of VOO (2018–2023).
- **Split**: 80% training, 20% testing (time-ordered).
- **Scaling**: StandardScaler fit on training data, applied to testing data.
- **Windowing**: Sequences generated post-split/scaling. Input shape: `(window_size, 1)`.

### Model Architecture
| Layer       | Parameter        |
|-------------|------------------|
| Input       | (window size, 1) |
| LSTM        |               x0 |
| LSTM (x1)   |               x2 |
| LSTM (x3)   |               x4 |
| Dense       |               x5 |
| Dropout     |               x6 |
| Dense       |               1  |

- **Dynamic Layers**:
  - Up to 3 LSTM layers (existence determined by ARO flags `x1`, `x3`).
  - Neurons per layer: Optimized within [1–20].
  - Dropout rate: Selected from `{0.3, 0.4, 0.5, 0.6, 0.7}`.

- **Hyperparameters**:
  - Optimizer: Chosen from `{Adagrad, Adam, Adamax, RMSprop, SGD}`.
  - Learning rate: Selected from `{1e-2, 1e-3, 1e-4, 1e-5, 1e-6}`.

### Hyperparameter Optimization with ARO
- **Fitness Function**: Validation MSE.
- **Energy-Based Switching**: Balances exploration and refinement using:
```python
energy = 4 * (1 - (iteration / max_iteration)) * np.log(1 / np.random.rand())
```
  - **Mutation**: Adjusts hyperparameters based on best candidate and energy levels.

## Results & Evaluation

### Metrics (Testing Only)
- **Primary Metrics**: MSE, RMSE, MAE, MAPE, R².

### Visualization:
- **Training Curves**: Loss (train/validation) vs. epochs.
- **Predictions**: Actual (black) vs. train-predicted (blue) vs. test-predicted (red).
- **Cross-Validation**: Grouped bar charts for metrics across folds.

### Example Output
- **Window Size**: 10
- **Best Hyperparameters**:

| Layer       | Parameter          |
|-------------|--------------------|
| Input       |            (10, 1) |
| LSTM        |                  9 |
| LSTM (0)    |                  0 |
| LSTM (1)    |                 15 |
| Dense       |                 14 |
| Dropout     | 0.3888546174385806 |
| Dense       |                  1 |
  - Optimizer: RMSprop
  - Learning Rate: 0.01

- **Test Metrics**:
  - MSE: 15475.887
  - RMSE: 124.402
  - MAE: 100.391
  - MAPE: 0.024
  - R²: 0.821

## Reproducibility Considerations
- **Randomness**: No fixed seeds set; results may vary slightly between runs.
- **Runtime**:
  - Main Program: ~1.5 minutes (RTX 2060 GPU).
  - Cross-Validation: ~2.8 minutes (3 folds per window).

## Citation
If using this work, cite the original paper:

- **LSTM-ARO**: Stock price prediction with optimized deep LSTM network with artificial rabbits optimization algorithm. [Link to paper](https://www.sciencedirect.com/science/article/pii/S0957417423008485)

## License
This code is provided "as-is" for academic and research purposes. Commercial use requires permission from the author.



