"""
ga_optimizer_v2.py -- Genetic Algorithm for Hyperparameter & Feature Optimization (V2)

Uses DEAP to co-optimize:
    - Feature selection (binary mask over technical indicators).
    - Model hyperparameters (window size, hidden units, dropout rate, learning rate, 
      threshold multiplier, confidence threshold).

Fitness Function:
    Maximize  Sharpe Ratio (simulated over Calibration set) - lambda * num_selected_features
"""

import logging
import pickle
import random
import gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from deap import base, creator, tools

from data_processor_v2 import DataProcessorV2
from model_builder import ClassificationBiLSTMModel, get_device
from trainer_v2 import TrainerV2
from confidence_model import train_confidence_model, extract_m2_features

logger = logging.getLogger(__name__)

# Hyperparameter options
WINDOW_OPTIONS = [6, 12, 24, 36]
HIDDEN_OPTIONS = [32, 64, 128, 256]
DROPOUT_OPTIONS = [0.1, 0.2, 0.3, 0.4]
LR_OPTIONS = [1e-3, 5e-4, 1e-4]
THRESH_MULT_OPTIONS = [0.3, 0.5, 0.7, 1.0]
CONF_THRESH_OPTIONS = [0.60, 0.65, 0.70, 0.75, 0.80]

NUM_HP_GENES = 6  # window, hidden, dropout, lr, thresh_mult, conf_thresh


class ChromosomeV2:
    def __init__(
        self,
        feature_mask: List[int],
        window_size: int,
        hidden_units: int,
        dropout: float,
        learning_rate: float,
        threshold_multiplier: float,
        confidence_threshold: float,
    ) -> None:
        self.feature_mask = feature_mask
        self.window_size = window_size
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.threshold_multiplier = threshold_multiplier
        self.confidence_threshold = confidence_threshold

    def to_dict(self, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        d = {
            "window_size": self.window_size,
            "hidden_units": self.hidden_units,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "threshold_multiplier": self.threshold_multiplier,
            "confidence_threshold": self.confidence_threshold,
            "num_selected_features": int(sum(self.feature_mask)),
        }
        if feature_names:
            d["selected_features"] = [n for n, b in zip(feature_names, self.feature_mask) if b == 1]
        return d

    @staticmethod
    def from_vector(vector: list, num_features: int) -> "ChromosomeV2":
        f_mask = [int(round(v)) for v in vector[:num_features]]
        start = num_features
        w_idx = int(round(vector[start]))
        h_idx = int(round(vector[start + 1]))
        d_idx = int(round(vector[start + 2]))
        l_idx = int(round(vector[start + 3]))
        tm_idx = int(round(vector[start + 4]))
        ct_idx = int(round(vector[start + 5]))

        return ChromosomeV2(
            feature_mask=f_mask,
            window_size=WINDOW_OPTIONS[min(w_idx, len(WINDOW_OPTIONS)-1)],
            hidden_units=HIDDEN_OPTIONS[min(h_idx, len(HIDDEN_OPTIONS)-1)],
            dropout=DROPOUT_OPTIONS[min(d_idx, len(DROPOUT_OPTIONS)-1)],
            learning_rate=LR_OPTIONS[min(l_idx, len(LR_OPTIONS)-1)],
            threshold_multiplier=THRESH_MULT_OPTIONS[min(tm_idx, len(THRESH_MULT_OPTIONS)-1)],
            confidence_threshold=CONF_THRESH_OPTIONS[min(ct_idx, len(CONF_THRESH_OPTIONS)-1)],
        )


class GAOptimizerV2:
    def __init__(
        self,
        feature_names: List[str],
        train_dfs: Dict[str, "pd.DataFrame"],
        cal_dfs: Dict[str, "pd.DataFrame"],
        population_size: int = 20,
        num_generations: int = 15,
        complexity_penalty: float = 0.05,
        crossover_prob: float = 0.7,
        mutation_prob: float = 0.2,
        ga_epochs: int = 5,
        result_logger: Optional["ResultLogger"] = None,
    ) -> None:
        self.feature_names = list(feature_names)
        self.num_features = len(feature_names)
        self.train_dfs = train_dfs
        self.cal_dfs = cal_dfs
        self.population_size = population_size
        self.num_generations = num_generations
        self.complexity_penalty = complexity_penalty
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.ga_epochs = ga_epochs
        self.result_logger = result_logger
        self.device = get_device()
        self.dp = DataProcessorV2()
        
        self._setup_deap_toolbox()

    def _setup_deap_toolbox(self) -> None:
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "IndividualV2"):
            creator.create("IndividualV2", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        self.toolbox.register("feature_bit", random.randint, 0, 1)
        self.toolbox.register("w_gene", random.randint, 0, len(WINDOW_OPTIONS)-1)
        self.toolbox.register("h_gene", random.randint, 0, len(HIDDEN_OPTIONS)-1)
        self.toolbox.register("d_gene", random.randint, 0, len(DROPOUT_OPTIONS)-1)
        self.toolbox.register("l_gene", random.randint, 0, len(LR_OPTIONS)-1)
        self.toolbox.register("tm_gene", random.randint, 0, len(THRESH_MULT_OPTIONS)-1)
        self.toolbox.register("ct_gene", random.randint, 0, len(CONF_THRESH_OPTIONS)-1)

        def _create_ind():
            genes = [self.toolbox.feature_bit() for _ in range(self.num_features)]
            genes.extend([
                self.toolbox.w_gene(), self.toolbox.h_gene(), self.toolbox.d_gene(),
                self.toolbox.l_gene(), self.toolbox.tm_gene(), self.toolbox.ct_gene()
            ])
            if sum(genes[:self.num_features]) < 3:
                for idx in random.sample(range(self.num_features), 3): genes[idx] = 1
            return creator.IndividualV2(genes)

        self.toolbox.register("individual", _create_ind)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.evaluate_individual)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", tools.cxTwoPoint)

        def _mutate(ind):
            for i in range(self.num_features):
                if random.random() < self.mutation_prob: ind[i] = 1 - ind[i]
            for j, max_v in enumerate([len(WINDOW_OPTIONS)-1, len(HIDDEN_OPTIONS)-1, 
                                       len(DROPOUT_OPTIONS)-1, len(LR_OPTIONS)-1, 
                                       len(THRESH_MULT_OPTIONS)-1, len(CONF_THRESH_OPTIONS)-1]):
                if random.random() < self.mutation_prob: ind[self.num_features + j] = random.randint(0, max_v)
            if sum(ind[:self.num_features]) < 3:
                for idx in random.sample(range(self.num_features), 3): ind[idx] = 1
            return (ind,)
            
        self.toolbox.register("mutate", _mutate)

    def log_generation(self, gen: int, record: dict, best_dict: dict) -> None:
        """Log generation stats and best parameters if a ResultLogger is attached."""
        if not self.result_logger:
            return
        
        run_dir = Path(self.result_logger.get_run_dir())
        
        # 1. Log fitness stats
        history_path = run_dir / "ga_history.csv"
        file_exists = history_path.exists()
        with open(history_path, "a", encoding="utf-8") as f:
            if not file_exists:
                f.write("generation,min_fitness,avg_fitness,max_fitness,std_fitness\n")
            f.write(f"{gen},{record['min']},{record['avg']},{record['max']},{record['std']}\n")
            
        # 2. Log best parameters
        params_path = run_dir / "ga_best_params_history.jsonl"
        with open(params_path, "a", encoding="utf-8") as f:
            import json
            log_obj = {"generation": gen, **best_dict}
            f.write(json.dumps(log_obj) + "\n")

    def compute_sharpe_fitness(self, model1, model2, cal_loader, conf_threshold) -> float:
        """Simulate P&L using Model 1 and Model 2 on calibration set."""
        m2_features, _ = extract_m2_features(model1, cal_loader, self.device)
        conf_scores = model2.predict_proba(m2_features)[:, 1]
        
        # Approved trades mask
        approved_mask = conf_scores > conf_threshold
        
        if approved_mask.sum() < 10:
            return -999.0  # Penalize models that never trade
            
        # Get Model 1 predictions and actual forward returns to simulate P&L
        model1.eval()
        all_probs, all_ret = [], []
        with torch.no_grad():
            for X_b, _, ret_b in cal_loader:
                logits = model1(X_b.to(self.device))
                all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
                all_ret.append(ret_b.numpy())
                
        all_probs = np.vstack(all_probs)
        all_ret = np.concatenate(all_ret)
        preds = np.argmax(all_probs, axis=1)
        
        # Convert class to direction: 0(DOWN)->-1, 1(NEUTRAL)->0, 2(UP)->+1
        direction = np.where(preds == 2, 1, np.where(preds == 0, -1, 0))
        
        pnl = direction[approved_mask] * all_ret[approved_mask]
        
        if pnl.std() < 1e-9:
            return -999.0
            
        sharpe = pnl.mean() / pnl.std() * np.sqrt(252 * 13) # Ann. 30min bars
        return sharpe

    def evaluate_individual(self, individual: list) -> Tuple[float]:
        chrom = ChromosomeV2.from_vector(individual, self.num_features)
        active_features = [n for n, b in zip(self.feature_names, chrom.feature_mask) if b == 1]
        if not active_features: return (-999.0,)

        # Re-compute targets with GA's threshold multiplier
        mod_train_dfs, mod_cal_dfs = {}, {}
        for ticker in self.train_dfs:
            mod_train_dfs[ticker] = self.dp.compute_targets_and_labels(
                self.train_dfs[ticker].copy(), chrom.threshold_multiplier)
            mod_cal_dfs[ticker] = self.dp.compute_targets_and_labels(
                self.cal_dfs[ticker].copy(), chrom.threshold_multiplier)

        # Scale & Window
        arrays = self.dp.scale_and_window_multi(
            mod_train_dfs, mod_cal_dfs, {}, active_features, chrom.window_size
        )
        
        model = ClassificationBiLSTMModel(len(active_features), chrom.hidden_units, 2, chrom.dropout)
        trainer = TrainerV2(model, self.device, chrom.learning_rate)
        
        train_loader, cal_loader = trainer.create_dataloaders(
            arrays["X_train"], arrays["y_train"], arrays["ret_train"],
            arrays["X_cal"], arrays["y_cal"], arrays["ret_cal"]
        )
        
        self._current_ind += 1
        logger.info(
            "Evaluating Gen %d/%d | Ind %d/%d...",
            self._current_gen, self.num_generations,
            self._current_ind, self._total_inds
        )
        
        # Train Model 1
        best_train_loss, best_val_loss = trainer.train(train_loader, cal_loader, epochs=self.ga_epochs, verbose=False)
        
        # Train Model 2
        model2 = train_confidence_model(model, cal_loader, self.device)
        
        # Compute Sharpe
        sharpe = self.compute_sharpe_fitness(model, model2, cal_loader, chrom.confidence_threshold)
        
        logger.info(
            "  -> M1 TrainLoss: %.4f | M1 ValLoss: %.4f | M2 Sharpe: %.4f",
            best_train_loss, best_val_loss, sharpe
        )
        
        penalty = self.complexity_penalty * len(active_features)
        
        individual_stats = chrom.to_dict(self.feature_names)
        individual_stats.update({
            "generation": self._current_gen,
            "individual_id": self._current_ind,
            "m1_train_loss": float(best_train_loss),
            "m1_val_loss": float(best_val_loss),
            "m2_sharpe": float(sharpe),
            "penalty": float(penalty),
            "fitness": float(sharpe - penalty)
        })
        if self.result_logger:
            self.result_logger.log_population_individual(individual_stats)
        
        # Cleanup memory
        del model, model2, trainer, train_loader, cal_loader, arrays
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Complexity Penalty
        penalty = self.complexity_penalty * len(active_features)
        return (sharpe - penalty,)

    def run(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("min", np.min)
        stats.register("avg", np.mean)
        stats.register("max", np.max)
        stats.register("std", np.std)

        start_gen, pop, hof = self._maybe_load_checkpoint(checkpoint_path)

        if start_gen == 0:
            pop = self.toolbox.population(n=self.population_size)
            hof = tools.HallOfFame(1)

            logger.info("=" * 60)
            logger.info("GA V2 Started -- Pop: %d, Gens: %d", self.population_size, self.num_generations)

            self._current_gen = 0
            self._current_ind = 0
            self._total_inds = len(pop)
            fitnesses = list(map(self.toolbox.evaluate, pop))
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit
            hof.update(pop)
            
            record = stats.compile(pop)
            
            # Get best individual for parameter logging
            best_ind = tools.selBest(pop, 1)[0]
            best_chrom = ChromosomeV2.from_vector(best_ind, self.num_features)
            best_dict = best_chrom.to_dict(self.feature_names)
            
            self.log_generation(0, record, best_dict)
            self._save_checkpoint(checkpoint_path, 0, pop, hof)
            start_gen = 1

        for gen in range(start_gen, self.num_generations + 1):
            offspring = self.toolbox.select(pop, len(pop))
            offspring = list(map(self.toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < self.mutation_prob:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            
            self._current_gen = gen
            self._current_ind = 0
            self._total_inds = len(invalid_ind)
            fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            pop[:] = offspring
            hof.update(pop)
            
            record = stats.compile(pop)
            
            # Get best individual for parameter logging
            best_ind_gen = tools.selBest(pop, 1)[0]
            best_chrom_gen = ChromosomeV2.from_vector(best_ind_gen, self.num_features)
            best_dict_gen = best_chrom_gen.to_dict(self.feature_names)
            
            self.log_generation(gen, record, best_dict_gen)
            self._save_checkpoint(checkpoint_path, gen, pop, hof)
            
            best_so_far = hof[0].fitness.values[0]
            logger.info("Gen %d complete. Best Sharpe so far: %.4f", gen, best_so_far)

        best_ind = hof[0]
        best_chrom = ChromosomeV2.from_vector(best_ind, self.num_features)
        best_dict = best_chrom.to_dict(self.feature_names)

        logger.info("GA V2 Complete! Best Sharpe: %.4f", best_ind.fitness.values[0])
        logger.info("Best config: %s", best_dict)
        
        if self.result_logger:
            self.result_logger.log_best_chromosome(best_dict)
        
        return {"best_chromosome": best_dict, "best_fitness": best_ind.fitness.values[0]}

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self,
        path: Optional[str],
        generation: int,
        pop: list,
        hof: tools.HallOfFame,
    ) -> None:
        if path is None:
            return

        ckpt = {
            "generation": generation,
            "population": [list(ind) for ind in pop],
            "fitnesses": [ind.fitness.values for ind in pop],
            "hof": [list(ind) for ind in hof],
            "hof_fitnesses": [ind.fitness.values for ind in hof],
            "random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "torch_cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "run_dir": str(self.result_logger.get_run_dir()) if self.result_logger else None,
        }

        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(ckpt, f)

        logger.info("[CHECKPOINT] Saved V2 GA state after gen %d to %s", generation, filepath)

    def _maybe_load_checkpoint(
        self,
        path: Optional[str],
    ) -> Tuple[int, Optional[list], Optional[tools.HallOfFame]]:
        if path is None or not Path(path).exists():
            return 0, None, None

        with open(path, "rb") as f:
            ckpt = pickle.load(f)

        random.setstate(ckpt["random_state"])
        np.random.set_state(ckpt["numpy_random_state"])
        
        if "torch_random_state" in ckpt:
            torch.set_rng_state(ckpt["torch_random_state"])
        if ckpt.get("torch_cuda_random_state") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ckpt["torch_cuda_random_state"])
            
        run_dir = ckpt.get("run_dir")
        if run_dir and Path(run_dir).exists() and self.result_logger:
            old_dir = self.result_logger.run_dir
            if old_dir != Path(run_dir):
                self.result_logger.run_dir = Path(run_dir)
                self.result_logger._ga_history_path = self.result_logger.run_dir / "ga_history.csv"
                self.result_logger._training_log_path = self.result_logger.run_dir / "training_log.csv"
                if old_dir.exists() and not any(old_dir.iterdir()):
                    old_dir.rmdir()

        pop = []
        for genes, fit in zip(ckpt["population"], ckpt["fitnesses"]):
            ind = creator.IndividualV2(genes)
            ind.fitness.values = fit
            pop.append(ind)

        hof = tools.HallOfFame(1)
        for genes, fit in zip(ckpt["hof"], ckpt["hof_fitnesses"]):
            ind = creator.IndividualV2(genes)
            ind.fitness.values = fit
            hof.update([ind])

        last_gen = ckpt["generation"]
        logger.info("[CHECKPOINT] Resumed V2 GA from gen %d / %d (file: %s)", last_gen, self.num_generations, path)
        return last_gen + 1, pop, hof
