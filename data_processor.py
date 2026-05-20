"""
data_processor.py — Data Ingestion, Preprocessing & Windowing

Handles:
    - Loading raw 5-minute OHLCV data for the 20 target companies.
    - Handling missing intervals (forward fill, then backward fill).
    - Computing multi-horizon target variables (Rate of Return).
    - Strict chronological Train/Validation/Test splitting.
    - Scaling features (fitted ONLY on the training set).
    - Sliding-window transformation into 3D tensors
      [Samples, Time_Window, Features].
"""

import logging
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


logger = logging.getLogger(__name__)


class DataProcessor:
    """Central class for all data loading, cleaning, splitting, scaling,
    and windowing operations.

    Attributes:
        scaler: Fitted scaler instance (StandardScaler or MinMaxScaler).
        feature_columns: List of feature column names after engineering.
        target_columns: List of target column names.
    """

    def __init__(
        self,
        scaler_type: str = "standard",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> None:
        """Initialise the DataProcessor.

        Args:
            scaler_type: One of 'standard' or 'minmax'.
            train_ratio: Fraction of data for training (default 0.8).
            val_ratio: Fraction of data for validation (default 0.1).
            test_ratio: Fraction of data for testing (default 0.1).
        """
        pass

    # ------------------------------------------------------------------
    # Data Loading
    # ------------------------------------------------------------------

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load raw OHLCV data from a CSV file.

        Args:
            filepath: Path to the CSV file containing 5-minute OHLCV data.

        Returns:
            DataFrame with parsed datetime index and OHLCV columns.
        """
        pass

    def handle_missing_intervals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill gaps in the time series.

        Applies forward fill first, then backward fill to handle any
        remaining leading NaNs.

        Args:
            df: Raw DataFrame potentially containing missing intervals.

        Returns:
            DataFrame with no missing values.
        """
        pass

    # ------------------------------------------------------------------
    # Target Variable Generation
    # ------------------------------------------------------------------

    def compute_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute multi-horizon Rate of Return target variables.

        Generates:
            - Target_1_Tick: (Close[t+1] - Close[t]) / Close[t]
            - Target_2_Tick: (Close[t+2] - Close[t]) / Close[t]
            - Target_3_Tick: (Close[t+3] - Close[t]) / Close[t]

        Args:
            df: DataFrame that must contain a 'Close' column.

        Returns:
            DataFrame with three new target columns appended.
        """
        pass

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def chronological_split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data strictly chronologically (no shuffling).

        Args:
            df: Full DataFrame ordered by time.

        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        pass

    # ------------------------------------------------------------------
    # Scaling
    # ------------------------------------------------------------------

    def fit_scaler(self, train_df: pd.DataFrame, feature_cols: List[str]) -> None:
        """Fit the scaler on the training set ONLY.

        Args:
            train_df: Training partition of the data.
            feature_cols: Column names to scale.
        """
        pass

    def transform(self, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        """Apply the already-fitted scaler to any partition.

        Args:
            df: DataFrame partition (train, val, or test).
            feature_cols: Column names to transform.

        Returns:
            Scaled numpy array of shape [n_samples, n_features].
        """
        pass

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Reverse scaling to original value space.

        Args:
            data: Scaled numpy array.

        Returns:
            Array in original scale.
        """
        pass

    # ------------------------------------------------------------------
    # Windowing
    # ------------------------------------------------------------------

    def create_windows(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        window_size: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transform 2D arrays into 3D sliding-window tensors.

        Args:
            features: Scaled feature array [n_samples, n_features].
            targets: Target array [n_samples,] or [n_samples, n_targets].
            window_size: Number of time steps per sample (look-back).

        Returns:
            Tuple of:
                X — shape [n_windows, window_size, n_features]
                y — shape [n_windows,] or [n_windows, n_targets]
        """
        pass
