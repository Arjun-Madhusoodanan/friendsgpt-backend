# generate_character_cache.py

import pandas as pd
import pickle
from friendsgpt_engine import load_data, compute_character_stats

if __name__ == "__main__":
    print("Loading data...")
    main_character_dialogues, _ = load_data()

    print("Computing character stats...")
    stats = compute_character_stats(main_character_dialogues)

    with open("cached_character_stats.pkl", "wb") as f:
        pickle.dump(stats, f)

    print("✅ Cache saved as cached_character_stats.pkl")
