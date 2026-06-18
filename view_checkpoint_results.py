"""
view_checkpoint_results.py

Utility script to load a DEAP genetic algorithm checkpoint and print the
best parameter configuration found up to that generation.
"""

import pickle
from pathlib import Path
from deap import creator, base
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ga_full_v2_multi_checkpoint.pkl",
                        help="Path to the .pkl checkpoint file")
    args = parser.parse_args()

    path = Path(args.checkpoint)
    if not path.exists():
        print(f"Error: Checkpoint file not found at {path}")
        return

    # DEAP needs these classes defined to unpickle the individuals
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "IndividualV2"):
        creator.create("IndividualV2", list, fitness=creator.FitnessMax)

    with open(path, "rb") as f:
        ckpt = pickle.load(f)

    print("=" * 60)
    print(f"CHECKPOINT LOADED: {path.name}")
    print(f"Generation Reached: {ckpt['generation']}")
    print("=" * 60)

    # hof[0] is the absolute best individual found so far across all generations
    best_genes = ckpt["hof"][0]
    best_fitness = ckpt["hof_fitnesses"][0][0]

    print(f"\nBest Sharpe Ratio (Fitness) so far: {best_fitness:.4f}")
    
    # We can rebuild the configuration using our options logic from ga_optimizer_v2
    # We will just print the raw genes, but also try to decode them if possible.
    print("\nBest Chromosome Vector (Raw Genes):")
    print(best_genes)

    # If you want to temporarily train on these parameters without running the GA,
    # you can manually plug these values into baseline_v2_multi in main.py!

    print("\nTo continue the full GA run, simply run:")
    print("python main.py ga_v2_multi")

if __name__ == "__main__":
    main()
