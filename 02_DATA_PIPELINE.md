# Data & Feature Engineering Pipeline (`02_DATA_PIPELINE.md`)

## 1. Base Data Loading
* Load 5-minute OHLCV data for the 20 target companies.
* Handle missing intervals methodically (forward fill, followed by backward fill).

## 2. Feature Engineering (Technical Indicators)
The agent should implement functions to generate the following features using the `ta` library or custom `pandas` calculations. 
*(Note: Fundamental metrics like P/E and P/B are excluded as they remain static across 5-minute intervals).*

### Volatility & Price Action
* **High-Low Spread:** ${High - Low} / {Low}$
* **Close-Open Spread:** ${Close - Open} / {Open}$
* **Bollinger Bands:** High, Low, and Mid bands.
* **ATR (Average True Range):** Crucial for capturing 5-minute volatility.

### Trend & Momentum
* **SMA (Simple Moving Average):** e.g., 10 and 20 periods.
* **EMA (Exponential Moving Average):** Preferred for high-frequency data.
* **MACD:** Both Line and Histogram.
* **RSI (Relative Strength Index)**

### Volume
* **VWAP (Volume Weighted Average Price)**
* **OBV (On-Balance Volume)**

## 3. Target Variable Generation
The model is **NOT** predicting the absolute price. It predicts the **Rate of Return**.

* **Target_1_Tick:** $({Close_{t+1} - Close_t}) / {Close_t}$
* **Target_2_Tick:** $({Close_{t+2} - Close_t}) / {Close_t}$
* **Target_3_Tick:** $({Close_{t+3} - Close_t}) / {Close_t}$

## 4. Scaling and Windowing (Time-Series Generation)
* **Scaling:** All features must be scaled using `StandardScaler` or `MinMaxScaler`. 
  > **Crucial Constraint:** Fit the scaler **ONLY** on the training set to definitively prevent data leakage.
* **Windowing:** Transform the 2D tabular data into 3D tensors: `[Samples, Time_Window, Features]`. The `Time_Window` must be configured as a dynamic variable that the Genetic Algorithm can modify during optimization.
