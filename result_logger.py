"""
result_logger.py -- Experiment Results Persistence

Creates a timestamped subfolder under RESULTS/ for every experiment run
and provides methods to log configuration, metrics, training history,
GA statistics, and model checkpoints.
"""

import csv
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

RESULTS_ROOT = Path("RESULTS")


class ResultLogger:
    """Manages a single experiment run's result folder and files."""

    def __init__(self, experiment_name: str, sub_label: str = "") -> None:
        """Create a timestamped run folder under RESULTS/.

        Args:
            experiment_name: Type of experiment (e.g., 'baseline', 'ga_features_only').
            sub_label: Additional label (e.g., ticker name 'jpm').
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parts = [experiment_name]
        if sub_label:
            parts.append(sub_label)
        parts.append(timestamp)
        folder_name = "_".join(parts)

        self.run_dir = RESULTS_ROOT / folder_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Prepare the training log CSV header
        self._training_log_path = self.run_dir / "training_log.csv"
        self._ga_history_path = self.run_dir / "ga_history.csv"
        self._training_log_initialized = False
        self._ga_history_initialized = False

        logger.info("ResultLogger created run folder: %s", self.run_dir)

    def get_run_dir(self) -> Path:
        """Return the absolute path of the current run folder."""
        return self.run_dir.resolve()

    def log_config(self, config: Dict[str, Any]) -> None:
        """Write experiment configuration to config.json.

        Args:
            config: Dictionary of all hyperparameters and settings.
        """
        path = self.run_dir / "config.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, default=str)
        logger.info("Config saved to %s", path)

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Write final evaluation metrics to metrics.json.

        Args:
            metrics: Dictionary of metric names to values.
        """
        path = self.run_dir / "metrics.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Metrics saved to %s", path)

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float) -> None:
        """Append one row to the training_log.csv.

        Args:
            epoch: Current epoch number.
            train_loss: Average training loss for this epoch.
            val_loss: Average validation loss for this epoch.
        """
        if not self._training_log_initialized:
            with open(self._training_log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_loss", "val_loss"])
            self._training_log_initialized = True

        with open(self._training_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.8f}", f"{val_loss:.8f}"])

    def log_ga_generation(self, gen: int, stats: Dict[str, float]) -> None:
        """Append one row to the ga_history.csv.

        Args:
            gen: Current generation number.
            stats: Dictionary with keys like 'min', 'avg', 'max', 'std'.
        """
        if not self._ga_history_initialized:
            with open(self._ga_history_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["generation", "min_fitness", "avg_fitness", "max_fitness", "std_fitness"])
            self._ga_history_initialized = True

        with open(self._ga_history_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                gen,
                f"{stats.get('min', 0):.8f}",
                f"{stats.get('avg', 0):.8f}",
                f"{stats.get('max', 0):.8f}",
                f"{stats.get('std', 0):.8f}",
            ])

    def log_best_chromosome(self, chromosome: Dict[str, Any]) -> None:
        """Write the best chromosome to best_chromosome.json.

        Args:
            chromosome: Decoded chromosome dictionary.
        """
        path = self.run_dir / "best_chromosome.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(chromosome, f, indent=2, default=str)
        logger.info("Best chromosome saved to %s", path)

    def copy_checkpoint(self, src_path: str) -> None:
        """Copy a model checkpoint file into the run folder.

        Args:
            src_path: Path to the source .pt checkpoint file.
        """
        src = Path(src_path)
        if not src.exists():
            logger.warning("Checkpoint file not found: %s", src)
            return
        dest = self.run_dir / "model.pt"
        shutil.copy2(src, dest)
        logger.info("Checkpoint copied to %s", dest)

    def log_population_individual(self, individual_stats: Dict[str, Any]) -> None:
        """Append one individual's stats to ga_population_history.jsonl."""
        path = self.run_dir / "ga_population_history.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(individual_stats, default=str) + "\n")

    def log_test_trades(self, df: Any) -> None: # Using Any to avoid pd import if not needed, but typical is pd.DataFrame
        """Save the detailed test trade log dataframe to CSV."""
        path = self.run_dir / "test_trade_log.csv"
        df.to_csv(path, index=False)
        logger.info("Test trade log saved to %s", path)

    def log_json_data(self, filename: str, data: Any) -> None:
        """Save a generic dictionary or list to a JSON file."""
        path = self.run_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("%s saved to %s", filename, path)
