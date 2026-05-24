"""
data_processor.py — Data Ingestion, Preprocessing & Windowing

Handles:
    - Loading clean 5-minute OHLCV data from CSV files.
      (Raw Refinitiv/LSEG .xlsx exports should first be converted
       using convert_xlsx_to_csv.py.)
    - Handling missing intervals (forward fill, then backward fill).
    - Computing multi-horizon target variables (Rate of Return).
    - Strict chronological Train/Validation/Test splitting.
    - Scaling features (fitted ONLY on the training set).
    - Sliding-window transformation into 3D tensors
      [Samples, Time_Window, Features].
"""

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from feature_engineer import FeatureEngineer

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


logger = logging.getLogger(__name__)

# Columns we keep from the raw Refinitiv export after parsing
_OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
_TARGET_COLUMNS = ["Target_1_Tick", "Target_2_Tick", "Target_3_Tick"]


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

        Raises:
            ValueError: If ratios do not sum to 1.0 or scaler_type is invalid.
        """
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError(
                f"Split ratios must sum to 1.0, got "
                f"{train_ratio + val_ratio + test_ratio:.4f}"
            )
        if scaler_type not in ("standard", "minmax"):
            raise ValueError(
                f"scaler_type must be 'standard' or 'minmax', got '{scaler_type}'"
            )

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        self._scaler_type = scaler_type
        self.scaler: Optional[StandardScaler | MinMaxScaler] = (
            StandardScaler() if scaler_type == "standard" else MinMaxScaler()
        )
        self._scaler_fitted = False
        self.feature_columns: List[str] = []
        self.target_columns: List[str] = list(_TARGET_COLUMNS)
        # Per-stock scalers for multi-asset mode
        self.stock_scalers: Dict[str, StandardScaler | MinMaxScaler] = {}

        logger.info(
            "DataProcessor initialised -- scaler=%s, split=%.0f/%.0f/%.0f",
            scaler_type,
            train_ratio * 100,
            val_ratio * 100,
            test_ratio * 100,
        )

    def _create_scaler(self) -> StandardScaler | MinMaxScaler:
        """Create a new scaler instance matching the configured type."""
        if self._scaler_type == "standard":
            return StandardScaler()
        return MinMaxScaler()

    # ------------------------------------------------------------------
    # Data Loading
    # ------------------------------------------------------------------

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load clean OHLCV data from a CSV file.

        Expects CSVs produced by ``convert_xlsx_to_csv.py`` with columns:
        Datetime, Open, High, Low, Close, Volume.  The Datetime column
        is parsed and set as a DatetimeIndex.

        Args:
            filepath: Path to the .csv file with 5-minute OHLCV data.

        Returns:
            DataFrame with DatetimeIndex ('Datetime') and columns
            ['Open', 'High', 'Low', 'Close', 'Volume'].
        """
        df = pd.read_csv(filepath, parse_dates=["Datetime"], index_col="Datetime")

        # Keep only OHLCV columns (safety check)
        df = df[_OHLCV_COLUMNS].copy()

        # Ensure chronological order
        df = df.sort_index(ascending=True)

        ticker = Path(filepath).stem
        logger.info(
            "Loaded %s -- %d rows, date range %s -> %s",
            ticker,
            len(df),
            df.index.min(),
            df.index.max(),
        )
        return df

    def load_all_data(self, data_dir: str) -> Dict[str, pd.DataFrame]:
        """Load every .csv file in a directory and return a dict of DataFrames.

        Args:
            data_dir: Path to the directory containing .csv files.

        Returns:
            Dictionary mapping ticker symbol (filename stem) to its
            cleaned DataFrame.
        """
        data_dir = Path(data_dir)
        datasets: Dict[str, pd.DataFrame] = {}

        csv_files = sorted(data_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No .csv files found in {data_dir}")

        for fpath in csv_files:
            ticker = fpath.stem
            try:
                df = self.load_data(str(fpath))
                datasets[ticker] = df
                logger.info("[OK] %s loaded (%d rows)", ticker, len(df))
            except Exception as exc:
                logger.error("[FAIL] Failed to load %s: %s", ticker, exc)

        logger.info("Loaded %d / %d files", len(datasets), len(csv_files))
        return datasets

    # ------------------------------------------------------------------
    # Missing Data Handling
    # ------------------------------------------------------------------

    def handle_missing_intervals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill gaps in the time series.

        Applies forward fill first, then backward fill to handle any
        remaining leading NaNs.  Logs the number of filled values.

        Args:
            df: Raw DataFrame potentially containing NaN rows.

        Returns:
            DataFrame with no missing values in OHLCV columns.
        """
        n_missing_before = df.isnull().sum().sum()

        df = df.ffill()   # Forward fill
        df = df.bfill()   # Backward fill for any remaining leading NaNs

        n_missing_after = df.isnull().sum().sum()
        if n_missing_before > 0:
            logger.info(
                "Filled %d missing values (%d remaining)",
                n_missing_before - n_missing_after,
                n_missing_after,
            )
        return df

    # ------------------------------------------------------------------
    # Target Variable Generation
    # ------------------------------------------------------------------

    def compute_targets(
        self, df: pd.DataFrame, horizons: List[int] = None
    ) -> pd.DataFrame:
        """Compute multi-horizon Rate of Return target variables.

        Generates:
            - Target_1_Tick: (Close[t+1] - Close[t]) / Close[t]
            - Target_2_Tick: (Close[t+2] - Close[t]) / Close[t]
            - Target_3_Tick: (Close[t+3] - Close[t]) / Close[t]

        Rows at the tail of the series where the target cannot be computed
        (due to the forward shift) are dropped.

        Args:
            df: DataFrame that must contain a 'Close' column.
            horizons: List of forward-looking tick offsets (default [1, 2, 3]).

        Returns:
            DataFrame with target columns appended and incomplete tail
            rows removed.
        """
        if horizons is None:
            horizons = [1, 2, 3]

        close = df["Close"]
        for h in horizons:
            col_name = f"Target_{h}_Tick"
            # shift(-h) moves future Close values back to align with current row
            df[col_name] = (close.shift(-h) - close) / close

        # Drop rows where any target is NaN (the last `max(horizons)` rows)
        n_before = len(df)
        df = df.dropna(subset=[f"Target_{h}_Tick" for h in horizons])
        n_dropped = n_before - len(df)

        logger.info(
            "Computed targets for horizons %s -- dropped %d tail rows",
            horizons,
            n_dropped,
        )
        return df

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def chronological_split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data strictly chronologically (no shuffling).

        The DataFrame must already be sorted by time (ascending).

        Args:
            df: Full DataFrame ordered by time.

        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        n = len(df)
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()

        logger.info(
            "Chronological split -- train=%d [%s -> %s], "
            "val=%d [%s -> %s], test=%d [%s -> %s]",
            len(train_df),
            train_df.index.min(),
            train_df.index.max(),
            len(val_df),
            val_df.index.min(),
            val_df.index.max(),
            len(test_df),
            test_df.index.min(),
            test_df.index.max(),
        )
        return train_df, val_df, test_df

    # ------------------------------------------------------------------
    # Scaling
    # ------------------------------------------------------------------

    def fit_scaler(self, train_df: pd.DataFrame, feature_cols: List[str]) -> None:
        """Fit the scaler on the training set ONLY.

        This prevents data leakage by ensuring validation/test statistics
        never influence the scaler.

        Args:
            train_df: Training partition of the data.
            feature_cols: Column names to scale.
        """
        self.feature_columns = list(feature_cols)
        self.scaler.fit(train_df[feature_cols].values)
        self._scaler_fitted = True
        logger.info("Scaler fitted on %d training samples, %d features",
                     len(train_df), len(feature_cols))

    def transform(self, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        """Apply the already-fitted scaler to any partition.

        Args:
            df: DataFrame partition (train, val, or test).
            feature_cols: Column names to transform.

        Returns:
            Scaled numpy array of shape [n_samples, n_features].

        Raises:
            RuntimeError: If the scaler has not been fitted yet.
        """
        if not self._scaler_fitted:
            raise RuntimeError(
                "Scaler has not been fitted. Call fit_scaler() first."
            )
        return self.scaler.transform(df[feature_cols].values)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Reverse scaling to original value space.

        Args:
            data: Scaled numpy array of shape [n_samples, n_features].

        Returns:
            Array in original scale.

        Raises:
            RuntimeError: If the scaler has not been fitted yet.
        """
        if not self._scaler_fitted:
            raise RuntimeError(
                "Scaler has not been fitted. Call fit_scaler() first."
            )
        return self.scaler.inverse_transform(data)

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

        For each index *i* from ``window_size`` to ``n_samples``, the
        window ``features[i - window_size : i]`` is paired with
        ``targets[i]`` (the value immediately after the window).

        Args:
            features: Scaled feature array [n_samples, n_features].
            targets: Target array [n_samples,] or [n_samples, n_targets].
            window_size: Number of time steps per sample (look-back).

        Returns:
            Tuple of:
                X — shape [n_windows, window_size, n_features]
                y — shape [n_windows,] or [n_windows, n_targets]
        """
        n_samples = features.shape[0]
        if window_size >= n_samples:
            raise ValueError(
                f"window_size ({window_size}) must be smaller than "
                f"n_samples ({n_samples})"
            )

        X: List[np.ndarray] = []
        y: List[np.ndarray] = []
        for i in range(window_size, n_samples):
            X.append(features[i - window_size : i])  # [window_size, n_features]
            y.append(targets[i])

        X_arr = np.array(X)  # [n_windows, window_size, n_features]
        y_arr = np.array(y)  # [n_windows,] or [n_windows, n_targets]

        logger.info(
            "Created %d windows -- X shape %s, y shape %s",
            len(X_arr),
            X_arr.shape,
            y_arr.shape,
        )
        return X_arr, y_arr

    # ------------------------------------------------------------------
    # Full Preprocessing Pipeline
    # ------------------------------------------------------------------

    def prepare_single_asset(
        self,
        filepath: str,
        feature_cols: List[str],
        target_col: str = "Target_1_Tick",
        window_size: int = 12,
    ) -> Dict[str, np.ndarray]:
        """End-to-end preprocessing for a single asset file.

        Convenience method that chains: load → fill → targets → split →
        scale → window.

        Args:
            filepath: Path to the .csv file.
            feature_cols: List of feature column names to use.
            target_col: Which target column to predict (default Target_1_Tick).
            window_size: Look-back window size (default 12).

        Returns:
            Dictionary with keys:
                'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test'
            All values are numpy arrays.
        """
        # 1. Load and clean
        df = self.load_data(filepath)
        df = self.handle_missing_intervals(df)

        # 2. Compute targets (caller is expected to have added features
        #    via FeatureEngineer before calling, or feature_cols must
        #    already be present in the loaded data)
        df = self.compute_targets(df)

        # 3. Verify requested feature columns exist
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise KeyError(
                f"Feature columns not found in DataFrame: {missing}"
            )

        # 4. Chronological split
        train_df, val_df, test_df = self.chronological_split(df)

        # 5. Fit scaler on training data ONLY, then transform all partitions
        self.fit_scaler(train_df, feature_cols)
        X_train_scaled = self.transform(train_df, feature_cols)
        X_val_scaled = self.transform(val_df, feature_cols)
        X_test_scaled = self.transform(test_df, feature_cols)

        # 6. Extract target arrays
        y_train = train_df[target_col].values
        y_val = val_df[target_col].values
        y_test = test_df[target_col].values

        # 7. Create sliding windows
        X_train, y_train = self.create_windows(X_train_scaled, y_train, window_size)
        X_val, y_val = self.create_windows(X_val_scaled, y_val, window_size)
        X_test, y_test = self.create_windows(X_test_scaled, y_test, window_size)

        logger.info(
            "Prepared asset %s -- train=%d, val=%d, test=%d windows",
            Path(filepath).stem,
            len(X_train),
            len(X_val),
            len(X_test),
        )

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
        }

    # ------------------------------------------------------------------
    # Multi-Asset Pipeline
    # ------------------------------------------------------------------

    def load_and_engineer_all(
        self,
        data_dir: str,
        feature_engineer: "FeatureEngineer",
    ) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
        """Load all stocks, clean, engineer features, compute targets.

        Args:
            data_dir: Directory containing .csv files.
            feature_engineer: FeatureEngineer instance.

        Returns:
            Tuple of (dict mapping ticker to prepared DataFrame, feature_cols).
        """
        raw_datasets = self.load_all_data(data_dir)
        prepared: Dict[str, pd.DataFrame] = {}
        feature_cols: Optional[List[str]] = None

        for ticker, df in raw_datasets.items():
            df = self.handle_missing_intervals(df)
            df = feature_engineer.add_all_features(df)
            if feature_cols is None:
                feature_cols = feature_engineer.get_feature_names()
            df = self.compute_targets(df)
            prepared[ticker] = df
            logger.info("[OK] %s fully prepared (%d rows)", ticker, len(df))

        logger.info(
            "Prepared %d stocks, %d features each",
            len(prepared),
            len(feature_cols or []),
        )
        return prepared, feature_cols or []

    def split_all_stocks(
        self,
        stock_dfs: Dict[str, pd.DataFrame],
    ) -> Tuple[
        Dict[str, pd.DataFrame],
        Dict[str, pd.DataFrame],
        Dict[str, pd.DataFrame],
    ]:
        """Apply per-stock chronological splits.

        Args:
            stock_dfs: Dict mapping ticker to its full DataFrame.

        Returns:
            Tuple of (train_dfs, val_dfs, test_dfs) dicts.
        """
        train_dfs: Dict[str, pd.DataFrame] = {}
        val_dfs: Dict[str, pd.DataFrame] = {}
        test_dfs: Dict[str, pd.DataFrame] = {}

        for ticker, df in stock_dfs.items():
            train, val, test = self.chronological_split(df)
            train_dfs[ticker] = train
            val_dfs[ticker] = val
            test_dfs[ticker] = test

        total_train = sum(len(d) for d in train_dfs.values())
        logger.info(
            "Split %d stocks -- total train rows: %d", len(stock_dfs), total_train
        )
        return train_dfs, val_dfs, test_dfs

    def scale_and_window_multi(
        self,
        train_dfs: Dict[str, pd.DataFrame],
        val_dfs: Dict[str, pd.DataFrame],
        test_dfs: Dict[str, pd.DataFrame],
        feature_cols: List[str],
        target_col: str = "Target_1_Tick",
        window_size: int = 12,
    ) -> Dict[str, np.ndarray]:
        """Per-stock scaling and boundary-safe windowing, then concatenation.

        Each stock gets its own scaler (fitted on training data only).
        Windows never cross stock boundaries.

        Args:
            train_dfs: Per-stock training DataFrames.
            val_dfs: Per-stock validation DataFrames.
            test_dfs: Per-stock test DataFrames.
            feature_cols: Feature column names to scale.
            target_col: Target column name.
            window_size: Sliding window size.

        Returns:
            Dict with keys X_train, y_train, X_val, y_val, X_test, y_test.
        """
        arrays = {k: [] for k in
                  ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]}

        for ticker in train_dfs:
            self._process_one_stock(
                ticker, train_dfs[ticker], val_dfs[ticker], test_dfs[ticker],
                feature_cols, target_col, window_size, arrays,
            )

        result = {k: np.concatenate(v) for k, v in arrays.items()}
        logger.info(
            "Multi-asset combined -- %d stocks, "
            "X_train=%s, X_val=%s, X_test=%s",
            len(train_dfs),
            result["X_train"].shape,
            result["X_val"].shape,
            result["X_test"].shape,
        )
        return result

    def _process_one_stock(
        self,
        ticker: str,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        window_size: int,
        arrays: Dict[str, List[np.ndarray]],
    ) -> None:
        """Scale and window a single stock, appending results to *arrays*."""
        scaler = self._create_scaler()
        scaler.fit(train_df[feature_cols].values)
        self.stock_scalers[ticker] = scaler

        for split_df, x_key, y_key in [
            (train_df, "X_train", "y_train"),
            (val_df, "X_val", "y_val"),
            (test_df, "X_test", "y_test"),
        ]:
            X_scaled = scaler.transform(split_df[feature_cols].values)
            y = split_df[target_col].values
            X_w, y_w = self.create_windows(X_scaled, y, window_size)
            arrays[x_key].append(X_w)
            arrays[y_key].append(y_w)
