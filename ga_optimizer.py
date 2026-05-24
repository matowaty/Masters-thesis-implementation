"""
ga_optimizer.py -- Genetic Algorithm for Hyperparameter & Feature Optimization

Uses DEAP to co-optimize:
    - Feature selection (binary mask over technical indicators).
    - Model hyperparameters (window size, look-forward horizon,
      hidden units, dropout rate, learning rate).

Chromosome Structure (flat vector):
    [feature_mask (N bits)] + [window_idx, horizon_idx, hidden_idx, dropout_idx, lr_idx]

Two modes:
    - "features_only": Only the feature mask mutates; HPs frozen at baseline.
    - "full": Both feature mask and HP genes are evolved.

Fitness Function:
    Minimize  RMSE_val + lambda * num_selected_features
"""

import logging
import random
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from deap import base, creator, tools

from data_processor import DataProcessor
from feature_engineer import FeatureEngineer
from model_builder import BiLSTMModel, get_device
from result_logger import ResultLogger
from trainer import Trainer

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Allowed hyperparameter values (indexed by integer gene)
# -----------------------------------------------------------------------
WINDOW_OPTIONS = [6, 12, 24, 36]
HORIZON_OPTIONS = [1, 2, 3]
HIDDEN_OPTIONS = [32, 64, 128, 256]
DROPOUT_OPTIONS = [0.1, 0.2, 0.3, 0.4]
LR_OPTIONS = [1e-3, 5e-4, 1e-4]

# Baseline defaults (used in "features_only" mode)
BASELINE_HP = {
    "window_idx": 1,    # 12
    "horizon_idx": 0,   # 1
    "hidden_idx": 1,    # 64
    "dropout_idx": 1,   # 0.2
    "lr_idx": 0,        # 1e-3
}

NUM_HP_GENES = 5  # window, horizon, hidden, dropout, lr


class Chromosome:
    """Represents a decoded GA individual."""

    def __init__(
        self,
        feature_mask: List[int],
        window_size: int,
        look_forward: int,
        hidden_units: int,
        dropout: float,
        learning_rate: float,
    ) -> None:
        self.feature_mask = feature_mask
        self.window_size = window_size
        self.look_forward = look_forward
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.learning_rate = learning_rate

    def to_dict(self, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Serialize the chromosome to a human-readable dictionary."""
        d = {
            "window_size": self.window_size,
            "look_forward": self.look_forward,
            "hidden_units": self.hidden_units,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "num_selected_features": int(sum(self.feature_mask)),
            "feature_mask": [int(b) for b in self.feature_mask],
        }
        if feature_names is not None:
            d["selected_features"] = [
                name for name, bit in zip(feature_names, self.feature_mask) if bit == 1
            ]
        return d

    @staticmethod
    def from_vector(vector: list, num_features: int) -> "Chromosome":
        """Reconstruct a Chromosome from a flat DEAP individual vector."""
        feature_mask = [int(round(v)) for v in vector[:num_features]]
        hp_start = num_features
        window_idx = int(round(vector[hp_start]))
        horizon_idx = int(round(vector[hp_start + 1]))
        hidden_idx = int(round(vector[hp_start + 2]))
        dropout_idx = int(round(vector[hp_start + 3]))
        lr_idx = int(round(vector[hp_start + 4]))

        return Chromosome(
            feature_mask=feature_mask,
            window_size=WINDOW_OPTIONS[min(window_idx, len(WINDOW_OPTIONS) - 1)],
            look_forward=HORIZON_OPTIONS[min(horizon_idx, len(HORIZON_OPTIONS) - 1)],
            hidden_units=HIDDEN_OPTIONS[min(hidden_idx, len(HIDDEN_OPTIONS) - 1)],
            dropout=DROPOUT_OPTIONS[min(dropout_idx, len(DROPOUT_OPTIONS) - 1)],
            learning_rate=LR_OPTIONS[min(lr_idx, len(LR_OPTIONS) - 1)],
        )


class GAOptimizer:
    """Wraps the DEAP evolutionary loop for hyperparameter search.

    Supports both single-stock mode (train_df/val_df) and multi-stock
    mode (train_dfs/val_dfs dicts with per-stock DataFrames).

    Args:
        feature_names: List of all available feature column names.
        train_df: Pre-split training DataFrame (single-stock mode).
        val_df: Pre-split validation DataFrame (single-stock mode).
        train_dfs: Dict of per-stock training DataFrames (multi-stock mode).
        val_dfs: Dict of per-stock validation DataFrames (multi-stock mode).
        mode: "features_only" or "full".
        population_size: Individuals per generation.
        num_generations: Max generations.
        complexity_penalty: Lambda for fitness penalty on feature count.
        crossover_prob: Crossover probability.
        mutation_prob: Per-gene mutation probability.
        ga_epochs: Epochs to train each individual during fitness evaluation.
        data_fraction: Fraction of data to use (1.0 = full data).
    """

    def __init__(
        self,
        feature_names: List[str],
        train_df: Optional["pd.DataFrame"] = None,
        val_df: Optional["pd.DataFrame"] = None,
        *,
        train_dfs: Optional[Dict[str, "pd.DataFrame"]] = None,
        val_dfs: Optional[Dict[str, "pd.DataFrame"]] = None,
        mode: str = "features_only",
        population_size: int = 20,
        num_generations: int = 30,
        complexity_penalty: float = 0.01,
        crossover_prob: float = 0.7,
        mutation_prob: float = 0.2,
        ga_epochs: int = 5,
        data_fraction: float = 1.0,
        result_logger: Optional[ResultLogger] = None,
    ) -> None:
        self.feature_names = list(feature_names)
        self.num_features = len(feature_names)
        self.mode = mode
        self.population_size = population_size
        self.num_generations = num_generations
        self.complexity_penalty = complexity_penalty
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.ga_epochs = ga_epochs
        self.result_logger = result_logger
        self.device = get_device()

        # Determine single-stock vs multi-stock mode
        self.multi_stock = train_dfs is not None

        if self.multi_stock:
            if data_fraction < 1.0:
                self.train_dfs = {
                    t: df.iloc[:int(len(df) * data_fraction)].copy()
                    for t, df in train_dfs.items()
                }
                self.val_dfs = {
                    t: df.iloc[:int(len(df) * data_fraction)].copy()
                    for t, df in val_dfs.items()
                }
                logger.info(
                    "GA multi-stock using %.0f%% data subset (%d stocks)",
                    data_fraction * 100, len(self.train_dfs),
                )
            else:
                self.train_dfs = {t: df.copy() for t, df in train_dfs.items()}
                self.val_dfs = {t: df.copy() for t, df in val_dfs.items()}
            # Not used in multi-stock mode
            self.train_df = None
            self.val_df = None
        else:
            # Single-stock mode (original behavior)
            if data_fraction < 1.0:
                n_train = int(len(train_df) * data_fraction)
                n_val = int(len(val_df) * data_fraction)
                self.train_df = train_df.iloc[:n_train].copy()
                self.val_df = val_df.iloc[:n_val].copy()
                logger.info(
                    "GA using %.0f%% data subset: train=%d, val=%d",
                    data_fraction * 100, n_train, n_val,
                )
            else:
                self.train_df = train_df.copy()
                self.val_df = val_df.copy()
            self.train_dfs = None
            self.val_dfs = None

        self.gene_length = self.num_features + NUM_HP_GENES
        self._setup_deap_toolbox()

    def _setup_deap_toolbox(self) -> None:
        """Register DEAP creator, individual, population, and operators."""
        # Avoid duplicate creator registration across multiple runs
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()

        # Gene generators
        self.toolbox.register("feature_bit", random.randint, 0, 1)
        self.toolbox.register("window_gene", random.randint, 0, len(WINDOW_OPTIONS) - 1)
        self.toolbox.register("horizon_gene", random.randint, 0, len(HORIZON_OPTIONS) - 1)
        self.toolbox.register("hidden_gene", random.randint, 0, len(HIDDEN_OPTIONS) - 1)
        self.toolbox.register("dropout_gene", random.randint, 0, len(DROPOUT_OPTIONS) - 1)
        self.toolbox.register("lr_gene", random.randint, 0, len(LR_OPTIONS) - 1)

        def _create_individual():
            """Build a single individual vector."""
            genes = [self.toolbox.feature_bit() for _ in range(self.num_features)]

            if self.mode == "features_only":
                # Freeze HPs at baseline
                genes.extend([
                    BASELINE_HP["window_idx"],
                    BASELINE_HP["horizon_idx"],
                    BASELINE_HP["hidden_idx"],
                    BASELINE_HP["dropout_idx"],
                    BASELINE_HP["lr_idx"],
                ])
            else:
                genes.extend([
                    self.toolbox.window_gene(),
                    self.toolbox.horizon_gene(),
                    self.toolbox.hidden_gene(),
                    self.toolbox.dropout_gene(),
                    self.toolbox.lr_gene(),
                ])

            # Guarantee at least 3 features are selected
            feature_part = genes[:self.num_features]
            if sum(feature_part) < 3:
                indices = random.sample(range(self.num_features), 3)
                for idx in indices:
                    genes[idx] = 1

            return creator.Individual(genes)

        self.toolbox.register("individual", _create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Genetic operators
        self.toolbox.register("evaluate", self.evaluate_individual)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

        if self.mode == "features_only":
            # Only crossover/mutate the feature mask portion
            self.toolbox.register(
                "mate", self._cx_features_only
            )
            self.toolbox.register(
                "mutate", self._mut_features_only
            )
        else:
            self.toolbox.register("mate", tools.cxTwoPoint)
            self.toolbox.register(
                "mutate", self._mut_full
            )

    def _cx_features_only(self, ind1, ind2):
        """Two-point crossover on the feature mask only; HPs untouched."""
        n = self.num_features
        tools.cxTwoPoint(ind1[:n], ind2[:n])
        return ind1, ind2

    def _mut_features_only(self, individual):
        """Flip bits in feature mask only; HPs untouched."""
        for i in range(self.num_features):
            if random.random() < self.mutation_prob:
                individual[i] = 1 - individual[i]
        # Guarantee minimum features
        if sum(individual[:self.num_features]) < 3:
            indices = random.sample(range(self.num_features), 3)
            for idx in indices:
                individual[idx] = 1
        return (individual,)

    def _mut_full(self, individual):
        """Mutate both feature mask and HP genes."""
        # Feature bits
        for i in range(self.num_features):
            if random.random() < self.mutation_prob:
                individual[i] = 1 - individual[i]

        # HP genes (uniform random within allowed range)
        hp_start = self.num_features
        hp_maxes = [
            len(WINDOW_OPTIONS) - 1,
            len(HORIZON_OPTIONS) - 1,
            len(HIDDEN_OPTIONS) - 1,
            len(DROPOUT_OPTIONS) - 1,
            len(LR_OPTIONS) - 1,
        ]
        for j, max_val in enumerate(hp_maxes):
            if random.random() < self.mutation_prob:
                individual[hp_start + j] = random.randint(0, max_val)

        # Guarantee minimum features
        if sum(individual[:self.num_features]) < 3:
            indices = random.sample(range(self.num_features), 3)
            for idx in indices:
                individual[idx] = 1

        return (individual,)

    def evaluate_individual(self, individual: list) -> Tuple[float]:
        """Fitness function: decode, train a model, return penalized RMSE.

        Returns:
            Single-element tuple (fitness_value,) as required by DEAP.
        """
        chrom = Chromosome.from_vector(individual, self.num_features)

        # Select active features
        active_features = [
            name for name, bit in zip(self.feature_names, chrom.feature_mask) if bit == 1
        ]
        if len(active_features) == 0:
            return (999.0,)  # penalty for empty feature set

        target_col = f"Target_{chrom.look_forward}_Tick"

        try:
            if self.multi_stock:
                X_train_w, y_train_w, X_val_w, y_val_w = (
                    self._scale_and_window_multi(
                        active_features, target_col, chrom.window_size
                    )
                )
            else:
                X_train_w, y_train_w, X_val_w, y_val_w = (
                    self._scale_and_window_single(
                        active_features, target_col, chrom.window_size
                    )
                )

            fitness = self._train_and_score(
                X_train_w, y_train_w, X_val_w, y_val_w,
                active_features, chrom,
            )

        except Exception as exc:
            logger.warning("Individual evaluation failed: %s", exc)
            fitness = 999.0

        return (fitness,)

    def _scale_and_window_single(
        self,
        active_features: List[str],
        target_col: str,
        window_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Scale and window for single-stock mode."""
        dp = DataProcessor(scaler_type="standard")
        dp.fit_scaler(self.train_df, active_features)

        X_train_scaled = dp.transform(self.train_df, active_features)
        X_val_scaled = dp.transform(self.val_df, active_features)

        y_train = self.train_df[target_col].values
        y_val = self.val_df[target_col].values

        X_train_w, y_train_w = dp.create_windows(X_train_scaled, y_train, window_size)
        X_val_w, y_val_w = dp.create_windows(X_val_scaled, y_val, window_size)

        return X_train_w, y_train_w, X_val_w, y_val_w

    def _scale_and_window_multi(
        self,
        active_features: List[str],
        target_col: str,
        window_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Per-stock scaling and boundary-safe windowing for multi-stock mode."""
        from sklearn.preprocessing import StandardScaler

        X_trains, y_trains, X_vals, y_vals = [], [], [], []
        dp = DataProcessor(scaler_type="standard")

        for ticker in self.train_dfs:
            scaler = StandardScaler()
            scaler.fit(self.train_dfs[ticker][active_features].values)

            X_tr = scaler.transform(self.train_dfs[ticker][active_features].values)
            X_va = scaler.transform(self.val_dfs[ticker][active_features].values)
            y_tr = self.train_dfs[ticker][target_col].values
            y_va = self.val_dfs[ticker][target_col].values

            X_tr_w, y_tr_w = dp.create_windows(X_tr, y_tr, window_size)
            X_va_w, y_va_w = dp.create_windows(X_va, y_va, window_size)

            X_trains.append(X_tr_w)
            y_trains.append(y_tr_w)
            X_vals.append(X_va_w)
            y_vals.append(y_va_w)

        return (
            np.concatenate(X_trains), np.concatenate(y_trains),
            np.concatenate(X_vals), np.concatenate(y_vals),
        )

    def _train_and_score(
        self,
        X_train_w: np.ndarray,
        y_train_w: np.ndarray,
        X_val_w: np.ndarray,
        y_val_w: np.ndarray,
        active_features: List[str],
        chrom: Chromosome,
    ) -> float:
        """Build, train model, and return fitness (RMSE + complexity penalty)."""
        model = BiLSTMModel(
            input_size=len(active_features),
            hidden_size=chrom.hidden_units,
            num_layers=2,
            dropout=chrom.dropout,
        )

        trainer = Trainer(
            model=model,
            device=self.device,
            learning_rate=chrom.learning_rate,
            loss_fn="mse",
        )

        train_loader, val_loader = trainer.create_dataloaders(
            X_train_w, y_train_w, X_val_w, y_val_w, batch_size=64
        )

        best_val_loss = trainer.train(
            train_loader, val_loader,
            epochs=self.ga_epochs,
            patience=self.ga_epochs + 1,
            verbose=False,
        )

        rmse = np.sqrt(best_val_loss)
        penalty = self.complexity_penalty * sum(chrom.feature_mask)
        return rmse + penalty

    def run(self) -> Dict[str, Any]:
        """Execute the full evolutionary optimisation loop.

        Returns:
            Dictionary containing:
                - 'best_chromosome': Best Chromosome found (as dict).
                - 'best_fitness': Its fitness score.
                - 'history': Per-generation statistics.
        """
        pop = self.toolbox.population(n=self.population_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("min", np.min)
        stats.register("avg", np.mean)
        stats.register("max", np.max)
        stats.register("std", np.std)

        logger.info("=" * 60)
        logger.info(
            "GA Optimization started -- mode=%s, pop=%d, gens=%d, epochs/ind=%d",
            self.mode, self.population_size, self.num_generations, self.ga_epochs,
        )
        logger.info("=" * 60)

        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        hof.update(pop)

        record = stats.compile(pop)
        self.log_generation(0, record)

        # Evolution loop
        for gen in range(1, self.num_generations + 1):
            # Selection
            offspring = self.toolbox.select(pop, len(pop))
            offspring = list(map(self.toolbox.clone, offspring))

            # Crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Mutation
            for mutant in offspring:
                if random.random() < self.mutation_prob:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate individuals with invalidated fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Replace population
            pop[:] = offspring
            hof.update(pop)

            record = stats.compile(pop)
            self.log_generation(gen, record)

        # Decode the best individual
        best_ind = hof[0]
        best_chrom = Chromosome.from_vector(best_ind, self.num_features)
        best_dict = best_chrom.to_dict(feature_names=self.feature_names)
        best_fitness = best_ind.fitness.values[0]

        logger.info("=" * 60)
        logger.info("GA complete -- best fitness: %.6f", best_fitness)
        logger.info("Best chromosome: %s", best_dict)
        logger.info("=" * 60)

        if self.result_logger:
            self.result_logger.log_best_chromosome(best_dict)

        return {
            "best_chromosome": best_dict,
            "best_fitness": best_fitness,
        }

    def log_generation(self, gen: int, record: dict) -> None:
        """Log statistics for the current generation."""
        logger.info(
            "Gen %3d -- min=%.6f  avg=%.6f  max=%.6f  std=%.6f",
            gen, record["min"], record["avg"], record["max"], record["std"],
        )
        if self.result_logger:
            self.result_logger.log_ga_generation(gen, record)
