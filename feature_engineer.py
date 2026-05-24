"""
feature_engineer.py — Technical Indicator Feature Generation

Calculates technical indicators from raw OHLCV data using the ``pandas_ta``
library (pandas-ta-classic) or custom Pandas math.  Grouped into three families:

    1. **Volatility & Price Action** — High-Low Spread, Close-Open Spread,
       Bollinger Bands, ATR.
    2. **Trend & Momentum** — SMA, EMA, MACD, RSI.
    3. **Volume** — VWAP, OBV.
"""

import logging
from typing import List

import pandas as pd
import pandas_ta_classic as ta  # pandas-ta-classic (pip install pandas-ta-classic)


logger = logging.getLogger(__name__)

# Mapping from pandas-ta auto-generated column names to our clean convention.
# Populated dynamically during feature generation.
_RENAME_MAP = {
    "BBL_20_2.0": "BB_Low",
    "BBM_20_2.0": "BB_Mid",
    "BBU_20_2.0": "BB_High",
    "BBB_20_2.0": "_BB_bandwidth_drop",   # bandwidth — not needed
    "BBP_20_2.0": "_BB_percent_drop",     # %B — not needed
    "ATRr_14": "ATR",
    "MACD_12_26_9": "MACD_Line",
    "MACDh_12_26_9": "MACD_Histogram",
    "MACDs_12_26_9": "_MACD_signal_drop",  # signal line — not needed
    "RSI_14": "RSI",
    "OBV": "OBV",
    "SMA_10": "SMA_10",
    "SMA_20": "SMA_20",
    "EMA_10": "EMA_10",
    "EMA_20": "EMA_20",
}

# Features we actually keep (after renaming), in canonical order.
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
]

# Columns to drop (extra outputs from pandas-ta that we don't need).
_COLUMNS_TO_DROP = [
    "_BB_bandwidth_drop",
    "_BB_percent_drop",
    "_MACD_signal_drop",
]


class FeatureEngineer:
    """Generate and manage technical indicator features.

    Attributes:
        feature_names: Ordered list of all generated feature column names.
    """

    def __init__(self) -> None:
        """Initialise the FeatureEngineer."""
        self.feature_names: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute and append every technical indicator to the DataFrame.

        Calls each family-specific method in sequence, renames auto-generated
        columns, drops auxiliary columns, removes leading NaN rows introduced
        by rolling windows, and returns the augmented DataFrame.

        Args:
            df: DataFrame with OHLCV columns and a DatetimeIndex.

        Returns:
            DataFrame with all indicator columns appended and NaN rows dropped.
        """
        n_before = len(df)

        # --- Volatility & Price Action ---
        df = self.add_high_low_spread(df)
        df = self.add_close_open_spread(df)
        df = self.add_bollinger_bands(df)
        df = self.add_atr(df)

        # --- Trend & Momentum ---
        df = self.add_sma(df)
        df = self.add_ema(df)
        df = self.add_macd(df)
        df = self.add_rsi(df)

        # --- Volume ---
        df = self.add_vwap(df)
        df = self.add_obv(df)

        # --- Stationary Past Returns ---
        df = self.add_past_returns(df)

        # Rename pandas-ta auto-generated column names to our convention
        df = df.rename(columns=_RENAME_MAP)

        # Drop auxiliary columns we don't need
        cols_to_drop = [c for c in _COLUMNS_TO_DROP if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        # Drop rows with NaN (from rolling window warm-up periods)
        df = df.dropna()
        n_dropped = n_before - len(df)

        # Store the feature names (Exclude non-stationary absolute prices)
        base_features = ["Volume"]
        self.feature_names = base_features + [
            f for f in _ENGINEERED_FEATURES if f in df.columns
        ]

        logger.info(
            "Added all features -- %d indicator columns, "
            "dropped %d NaN warm-up rows, %d rows remaining",
            len(self.feature_names) - len(base_features),
            n_dropped,
            len(df),
        )
        return df

    def get_feature_names(self) -> List[str]:
        """Return an ordered list of all generated feature column names.

        Returns:
            List of feature column name strings.
        """
        return list(self.feature_names)

    # ------------------------------------------------------------------
    # Volatility & Price Action
    # ------------------------------------------------------------------

    def add_high_low_spread(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute (High - Low) / Low spread.

        Args:
            df: DataFrame with 'High' and 'Low' columns.

        Returns:
            DataFrame with 'HL_Spread' column appended.
        """
        df["HL_Spread"] = (df["High"] - df["Low"]) / df["Low"]
        logger.debug("Added HL_Spread")
        return df

    def add_close_open_spread(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute (Close - Open) / Open spread.

        Args:
            df: DataFrame with 'Close' and 'Open' columns.

        Returns:
            DataFrame with 'CO_Spread' column appended.
        """
        df["CO_Spread"] = (df["Close"] - df["Open"]) / df["Open"]
        logger.debug("Added CO_Spread")
        return df

    def add_bollinger_bands(
        self, df: pd.DataFrame, window: int = 20
    ) -> pd.DataFrame:
        """Add Bollinger Bands (High, Mid, Low).

        Uses pandas-ta: ``df.ta.bbands(length=window, std=2, append=True)``.
        Auto-generated columns are renamed later in ``add_all_features()``.

        Args:
            df: DataFrame with 'Close' column.
            window: Rolling window period (default 20).

        Returns:
            DataFrame with Bollinger Band columns appended.
        """
        df.ta.bbands(length=window, std=2, append=True)
        logger.debug("Added Bollinger Bands (window=%d)", window)
        return df

    def add_atr(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Compute Average True Range (ATR).

        Uses pandas-ta: ``df.ta.atr(length=window, append=True)``.

        Args:
            df: DataFrame with 'High', 'Low', 'Close' columns.
            window: ATR period (default 14).

        Returns:
            DataFrame with ATR column appended.
        """
        df.ta.atr(length=window, append=True)
        logger.debug("Added ATR (window=%d)", window)
        return df

    # ------------------------------------------------------------------
    # Trend & Momentum
    # ------------------------------------------------------------------

    def add_past_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute the past rate of return for multiple lookbacks.
        
        This makes the momentum explicitly visible to the model as stationary features.
        """
        for ticks in [1, 2, 5]:
            col_name = f"Past_Return_{ticks}_Tick"
            df[col_name] = (df["Close"] - df["Close"].shift(ticks)) / df["Close"].shift(ticks)
            
        logger.debug("Added Past Returns (1, 2, 5 ticks)")
        return df

    def add_sma(
        self, df: pd.DataFrame, periods: List[int] = None
    ) -> pd.DataFrame:
        """Add Simple Moving Average(s).

        Uses pandas-ta: ``df.ta.sma(length=p, append=True)``.

        Args:
            df: DataFrame with 'Close' column.
            periods: List of SMA window sizes (default [10, 20]).

        Returns:
            DataFrame with 'SMA_{p}' columns appended.
        """
        if periods is None:
            periods = [10, 20]

        for p in periods:
            df.ta.sma(length=p, append=True)

        logger.debug("Added SMA for periods %s", periods)
        return df

    def add_ema(
        self, df: pd.DataFrame, periods: List[int] = None
    ) -> pd.DataFrame:
        """Add Exponential Moving Average(s).

        Uses pandas-ta: ``df.ta.ema(length=p, append=True)``.

        Args:
            df: DataFrame with 'Close' column.
            periods: List of EMA window sizes (default [10, 20]).

        Returns:
            DataFrame with 'EMA_{p}' columns appended.
        """
        if periods is None:
            periods = [10, 20]

        for p in periods:
            df.ta.ema(length=p, append=True)

        logger.debug("Added EMA for periods %s", periods)
        return df

    def add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute MACD Line and MACD Histogram.

        Uses pandas-ta: ``df.ta.macd(fast=12, slow=26, signal=9, append=True)``.
        Auto-generated columns are renamed later in ``add_all_features()``.

        Args:
            df: DataFrame with 'Close' column.

        Returns:
            DataFrame with MACD columns appended.
        """
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        logger.debug("Added MACD (12/26/9)")
        return df

    def add_rsi(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Compute Relative Strength Index.

        Uses pandas-ta: ``df.ta.rsi(length=window, append=True)``.

        Args:
            df: DataFrame with 'Close' column.
            window: RSI period (default 14).

        Returns:
            DataFrame with RSI column appended.
        """
        df.ta.rsi(length=window, append=True)
        logger.debug("Added RSI (window=%d)", window)
        return df

    # ------------------------------------------------------------------
    # Volume
    # ------------------------------------------------------------------

    def add_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Volume Weighted Average Price with daily reset.

        VWAP is computed intraday: cumulative sums of (Typical Price × Volume)
        and Volume are reset at each market open (i.e. grouped by date).

        Args:
            df: DataFrame with 'High', 'Low', 'Close', 'Volume' columns
                and a DatetimeIndex.

        Returns:
            DataFrame with 'VWAP' column appended.
        """
        typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
        tp_vol = typical_price * df["Volume"]

        # Group by trading day and compute cumulative VWAP within each session
        trading_day = df.index.date
        cum_tp_vol = tp_vol.groupby(trading_day).cumsum()
        cum_vol = df["Volume"].groupby(trading_day).cumsum()

        # Avoid division by zero for periods with no volume
        df["VWAP"] = cum_tp_vol / cum_vol.replace(0, float("nan"))

        logger.debug("Added VWAP (daily reset)")
        return df

    def add_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute On-Balance Volume.

        Uses pandas-ta: ``df.ta.obv(append=True)``.

        Args:
            df: DataFrame with 'Close' and 'Volume' columns.

        Returns:
            DataFrame with 'OBV' column appended.
        """
        df.ta.obv(append=True)
        logger.debug("Added OBV")
        return df
