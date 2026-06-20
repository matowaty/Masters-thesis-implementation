"""
data_processor_v2.py — Data Ingestion & Market-Aware Preprocessing (V2)

Handles:
    - Loading 5-min OHLCV.
    - Aggregating to 30-min timeframe.
    - Computing Market Context Features (Index Return, Index Volatility).
    - Dynamic volatility thresholding & 3-Class label generation.
    - 3-Way Splitting: Train (60%), Calibration (20%), Test (20%).
    - Sliding-window tensor generation for Deep Learning.
"""

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

if TYPE_CHECKING:
    from feature_engineer_v2 import FeatureEngineerV2

logger = logging.getLogger(__name__)

_OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

class DataProcessorV2:
    def __init__(
        self,
        scaler_type: str = "standard",
        train_ratio: float = 0.6,
        cal_ratio: float = 0.2,
        test_ratio: float = 0.2,
        target_scaling_factor: float = 1.0,  # Unused for classification, kept for compatibility
    ) -> None:
        if not np.isclose(train_ratio + cal_ratio + test_ratio, 1.0):
            raise ValueError(f"Split ratios must sum to 1.0, got {train_ratio + cal_ratio + test_ratio:.4f}")
        
        self.train_ratio = train_ratio
        self.cal_ratio = cal_ratio
        self.test_ratio = test_ratio

        self._scaler_type = scaler_type
        self.scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
        self._scaler_fitted = False
        self.feature_columns: List[str] = []
        self.stock_scalers: Dict[str, StandardScaler | MinMaxScaler] = {}

    def _create_scaler(self) -> StandardScaler | MinMaxScaler:
        return StandardScaler() if self._scaler_type == "standard" else MinMaxScaler()

    def load_data(self, filepath: str) -> pd.DataFrame:
        df = pd.read_csv(filepath, parse_dates=["Datetime"], index_col="Datetime")
        df = df[_OHLCV_COLUMNS].copy().sort_index(ascending=True)
        return df

    def resample_ohlcv(self, df_5min: pd.DataFrame, period: str = '30min') -> pd.DataFrame:
        """Aggregate 5-min bars into larger timeframes."""
        df = df_5min.resample(period).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        return df

    def handle_missing_intervals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.ffill().bfill()
        return df

    def compute_market_context(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Compute the average return and volatility across all stocks per timestamp."""
        # Extract 'Close' prices for all stocks
        closes = {}
        for ticker, df in datasets.items():
            closes[ticker] = df['Close']
            
        close_df = pd.DataFrame(closes)
        
        # Calculate 1-bar return for every stock
        returns_df = close_df.pct_change()
        
        # Cross-sectional mean and std (across columns)
        market_df = pd.DataFrame({
            'Index_Return': returns_df.mean(axis=1),
            'Index_Volatility': returns_df.std(axis=1)
        })
        
        return market_df

    def compute_targets_and_labels(
        self, df: pd.DataFrame, threshold_multiplier: float = 0.5, window: int = 100
    ) -> pd.DataFrame:
        """Compute forward returns, dynamic thresholds, and 3-class labels."""
        # 1. Forward return for 1 bar (e.g. next 30-min)
        if 'fwd_return' not in df.columns:
            df['fwd_return'] = df['Close'].shift(-1) / df['Close'] - 1.0
        
        # 2. Dynamic volatility threshold
        if 'rolling_vol' not in df.columns:
            past_returns = df['Close'].pct_change()
            df['rolling_vol'] = past_returns.rolling(window).std().bfill()
            
        df['volatility_threshold'] = df['rolling_vol'] * threshold_multiplier
        
        # 3. 3-class Labels: 0=DOWN, 1=NEUTRAL, 2=UP
        df['Target_Class'] = np.where(
            df['fwd_return'] > df['volatility_threshold'], 2,
            np.where(df['fwd_return'] < -df['volatility_threshold'], 0, 1)
        )
        
        # Drop the last row which has NaN forward return if not already dropped
        if df['fwd_return'].isna().any():
            df = df.dropna(subset=['fwd_return'])
        return df

    def load_and_engineer_all(
        self,
        data_dir: str,
        feature_engineer: "FeatureEngineerV2",
        threshold_multiplier: float = 0.5,
        resample_period: str = "30min",
    ) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
        
        data_dir = Path(data_dir)
        csv_files = sorted(data_dir.glob("*.csv"))
        
        raw_resampled = {}
        
        # 1. Load, clean, and resample
        for fpath in csv_files:
            ticker = fpath.stem
            df = self.load_data(str(fpath))
            df = self.handle_missing_intervals(df)
            df_resampled = self.resample_ohlcv(df, resample_period)
            raw_resampled[ticker] = df_resampled
            
        # 2. Compute Market Context globally
        market_df = self.compute_market_context(raw_resampled)
        
        prepared: Dict[str, pd.DataFrame] = {}
        feature_cols: Optional[List[str]] = None
        
        # 3. Inject context, engineer features, compute targets
        for ticker, df in raw_resampled.items():
            # Align market context
            df = df.join(market_df, how='left')
            
            # Engineer features
            df = feature_engineer.add_all_features(df)
            if feature_cols is None:
                feature_cols = feature_engineer.get_feature_names()
                
            # Compute targets
            df = self.compute_targets_and_labels(df, threshold_multiplier)
            prepared[ticker] = df
            
        return prepared, feature_cols or []

    def temporal_split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        n = len(df)
        train_end = int(n * self.train_ratio)
        cal_end = train_end + int(n * self.cal_ratio)

        train_df = df.iloc[:train_end].copy()
        cal_df = df.iloc[train_end:cal_end].copy()
        test_df = df.iloc[cal_end:].copy()

        return train_df, cal_df, test_df

    def split_all_stocks(
        self, stock_dfs: Dict[str, pd.DataFrame]
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        train_dfs, cal_dfs, test_dfs = {}, {}, {}
        for ticker, df in stock_dfs.items():
            tr, ca, te = self.temporal_split(df)
            train_dfs[ticker] = tr
            cal_dfs[ticker] = ca
            test_dfs[ticker] = te
        return train_dfs, cal_dfs, test_dfs

    def create_windows(
        self, features: np.ndarray, targets: np.ndarray, fwd_returns: np.ndarray, 
        timestamps: np.ndarray, tickers: np.ndarray, window_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_samples = features.shape[0]
        X, y, ret, times, tks = [], [], [], [], []
        for i in range(window_size, n_samples):
            X.append(features[i - window_size : i])
            y.append(targets[i])
            ret.append(fwd_returns[i])
            times.append(timestamps[i])
            tks.append(tickers[i])
        return np.array(X), np.array(y), np.array(ret), np.array(times), np.array(tks)

    def scale_and_window_multi(
        self,
        train_dfs: Dict[str, pd.DataFrame],
        cal_dfs: Dict[str, pd.DataFrame],
        test_dfs: Dict[str, pd.DataFrame],
        feature_cols: List[str],
        window_size: int = 12,
    ) -> Dict[str, np.ndarray]:
        
        arrays = {k: [] for k in ["X_train", "y_train", "ret_train", "time_train", "ticker_train",
                                  "X_cal", "y_cal", "ret_cal", "time_cal", "ticker_cal",
                                  "X_test", "y_test", "ret_test", "time_test", "ticker_test"]}

        for ticker in train_dfs.keys():
            scaler = self._create_scaler()
            scaler.fit(train_dfs[ticker][feature_cols].values)
            self.stock_scalers[ticker] = scaler

            splits = [
                (train_dfs.get(ticker), "X_train", "y_train", "ret_train", "time_train", "ticker_train"),
                (cal_dfs.get(ticker) if cal_dfs else None, "X_cal", "y_cal", "ret_cal", "time_cal", "ticker_cal"),
                (test_dfs.get(ticker) if test_dfs else None, "X_test", "y_test", "ret_test", "time_test", "ticker_test"),
            ]
            for split_df, x_key, y_key, ret_key, time_key, ticker_key in splits:
                if split_df is None or split_df.empty:
                    continue
                X_scaled = scaler.transform(split_df[feature_cols].values)
                y_class = split_df["Target_Class"].values
                ret_fwd = split_df["fwd_return"].values
                timestamps = split_df.index.values
                tickers = np.full(len(split_df), ticker)
                
                X_w, y_w, ret_w, time_w, ticker_w = self.create_windows(
                    X_scaled, y_class, ret_fwd, timestamps, tickers, window_size
                )
                arrays[x_key].append(X_w)
                arrays[y_key].append(y_w)
                arrays[ret_key].append(ret_w)
                arrays[time_key].append(time_w)
                arrays[ticker_key].append(ticker_w)

        return {k: np.concatenate(v) if v else np.array([]) for k, v in arrays.items()}
