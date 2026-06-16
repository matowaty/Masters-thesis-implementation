"""
feature_engineer_v2.py — Technical Indicator Feature Generation (V2 Classification Pipeline)

Calculates technical indicators from raw OHLCV data using the pandas_ta
library or custom Pandas math. Extended for V2 to include:
    - 5 missing features (RSI div, Volume surprise, Bar range, Close pos, VWAP dist)
    - Cross-stock relative features utilizing Market Context features provided by DataProcessorV2.
"""

import logging
from typing import List

import pandas as pd
import pandas_ta_classic as ta

logger = logging.getLogger(__name__)

_RENAME_MAP = {
    "BBL_20_2.0": "BB_Low",
    "BBM_20_2.0": "BB_Mid",
    "BBU_20_2.0": "BB_High",
    "BBB_20_2.0": "_BB_bandwidth_drop",
    "BBP_20_2.0": "_BB_percent_drop",
    "ATRr_14": "ATR",
    "MACD_12_26_9": "MACD_Line",
    "MACDh_12_26_9": "MACD_Histogram",
    "MACDs_12_26_9": "_MACD_signal_drop",
    "RSI_14": "RSI",
    "OBV": "OBV",
    "SMA_10": "SMA_10",
    "SMA_20": "SMA_20",
    "EMA_10": "EMA_10",
    "EMA_20": "EMA_20",
}

_ENGINEERED_FEATURES = [
    "Past_Return_1_Tick",
    "Past_Return_2_Tick",
    "Past_Return_5_Tick",
    "HL_Spread",
    "CO_Spread",
    "BB_High",
    "BB_Mid",
    "BB_Low",
    "ATR",
    "SMA_10",
    "SMA_20",
    "EMA_10",
    "EMA_20",
    "MACD_Line",
    "MACD_Histogram",
    "RSI",
    "VWAP",
    "OBV",
    # --- New V2 Features ---
    "RSI_Div",
    "Volume_Surprise",
    "Bar_Range",
    "Close_Pos",
    "VWAP_Dist",
    # --- Cross-Stock Features ---
    "Index_Return",      # Passed from DataProcessorV2
    "Index_Volatility",  # Passed from DataProcessorV2
    "Rel_Return",
    "Rel_ZScore"
]

_COLUMNS_TO_DROP = [
    "_BB_bandwidth_drop",
    "_BB_percent_drop",
    "_MACD_signal_drop",
]


class FeatureEngineerV2:
    """Generate and manage technical indicator features for V2 Pipeline."""

    def __init__(self) -> None:
        self.feature_names: List[str] = []

    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        n_before = len(df)

        # 1. Volatility & Price Action
        df = self.add_high_low_spread(df)
        df = self.add_close_open_spread(df)
        df = self.add_bollinger_bands(df)
        df = self.add_atr(df)

        # 2. Trend & Momentum
        df = self.add_sma(df)
        df = self.add_ema(df)
        df = self.add_macd(df)
        df = self.add_rsi(df)

        # 3. Volume
        df = self.add_vwap(df)
        df = self.add_obv(df)

        # 4. Stationary Past Returns
        df = self.add_past_returns(df)

        # 5. New V2 Features
        df = self.add_v2_features(df)

        # 6. Cross-Stock Features (Requires Index_Return/Index_Volatility)
        df = self.add_cross_stock_features(df)

        # Rename & Drop
        df = df.rename(columns=_RENAME_MAP)
        cols_to_drop = [c for c in _COLUMNS_TO_DROP if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        # Drop rows with NaN
        df = df.dropna()
        n_dropped = n_before - len(df)

        # Store feature names
        base_features = ["Volume"]
        self.feature_names = base_features + [
            f for f in _ENGINEERED_FEATURES if f in df.columns
        ]

        logger.info(
            "Added all V2 features -- %d indicator columns, dropped %d NaN warm-up rows",
            len(self.feature_names) - len(base_features),
            n_dropped,
        )
        return df

    def get_feature_names(self) -> List[str]:
        return list(self.feature_names)

    # ------------------------------------------------------------------
    # Feature Families
    # ------------------------------------------------------------------

    def add_high_low_spread(self, df: pd.DataFrame) -> pd.DataFrame:
        df["HL_Spread"] = (df["High"] - df["Low"]) / df["Low"]
        return df

    def add_close_open_spread(self, df: pd.DataFrame) -> pd.DataFrame:
        df["CO_Spread"] = (df["Close"] - df["Open"]) / df["Open"]
        return df

    def add_bollinger_bands(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        df.ta.bbands(length=window, std=2, append=True)
        return df

    def add_atr(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        df.ta.atr(length=window, append=True)
        return df

    def add_past_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        for ticks in [1, 2, 5]:
            df[f"Past_Return_{ticks}_Tick"] = (df["Close"] - df["Close"].shift(ticks)) / df["Close"].shift(ticks)
        return df

    def add_sma(self, df: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
        if periods is None: periods = [10, 20]
        for p in periods: df.ta.sma(length=p, append=True)
        return df

    def add_ema(self, df: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
        if periods is None: periods = [10, 20]
        for p in periods: df.ta.ema(length=p, append=True)
        return df

    def add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        return df

    def add_rsi(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        df.ta.rsi(length=window, append=True)
        return df

    def add_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
        tp_vol = typical_price * df["Volume"]
        trading_day = df.index.date
        cum_tp_vol = tp_vol.groupby(trading_day).cumsum()
        cum_vol = df["Volume"].groupby(trading_day).cumsum()
        df["VWAP"] = cum_tp_vol / cum_vol.replace(0, float("nan"))
        return df

    def add_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        df.ta.obv(append=True)
        return df

    # ------------------------------------------------------------------
    # V2 Specific Additions
    # ------------------------------------------------------------------

    def add_v2_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # RSI Divergence
        # Note: RSI_14 is the pandas-ta auto-generated name before renaming
        if "RSI_14" in df.columns:
            df["RSI_Div"] = df["RSI_14"] - df["RSI_14"].shift(5)
        else:
            # Fallback if already renamed
            df["RSI_Div"] = df["RSI"] - df["RSI"].shift(5) if "RSI" in df.columns else 0

        # Volume Surprise
        vol_mean = df["Volume"].rolling(20).mean().replace(0, float("nan"))
        df["Volume_Surprise"] = df["Volume"] / vol_mean

        # Bar Range
        df["Bar_Range"] = (df["High"] - df["Low"]) / df["Close"]

        # Close Position in Bar
        df["Close_Pos"] = (df["Close"] - df["Low"]) / (df["High"] - df["Low"] + 1e-9)

        # VWAP Distance
        df["VWAP_Dist"] = (df["Close"] - df["VWAP"]) / df["Close"]

        return df

    def add_cross_stock_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Relies on data_processor_v2 injecting Index_Return and Index_Volatility
        if "Index_Return" in df.columns and "Past_Return_1_Tick" in df.columns:
            df["Rel_Return"] = df["Past_Return_1_Tick"] - df["Index_Return"]
            
            # Cross-stock Z-Score over a rolling 20 bars
            # (stock_return - universe_mean) / universe_std
            # We already have Index_Volatility as universe_std
            df["Rel_ZScore"] = (df["Past_Return_1_Tick"] - df["Index_Return"]) / (df["Index_Volatility"] + 1e-9)
        else:
            logger.warning("Index features not found. Cross-stock features set to 0.")
            df["Rel_Return"] = 0.0
            df["Rel_ZScore"] = 0.0
            
        return df
