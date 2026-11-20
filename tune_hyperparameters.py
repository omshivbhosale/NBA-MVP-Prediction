"""Choose LightGBM hyperparameters using only seasons up to 2024.

Rolling-origin validation: for each validation season from 2015 to 2024, train on all
earlier seasons and score on that season. The 2025 and 2026 seasons are never touched,
so they remain clean test seasons.

Run from the repository root:  python tune_hyperparameters.py
Writes: best_params.json
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

SEED = 42
N_TRIALS = 40
VALIDATION_SEASONS = range(2015, 2025)

stats = pd.read_csv("player_mvp_stats.csv").fillna(0)
for col in ["PTS", "AST", "TRB", "STL", "BLK", "W/L%", "MP", "FG%"]:
    stats[f"{col}_R"] = stats[col] / stats.groupby("Year")[col].transform("mean")

target = "Share"
not_features = ["Player", "Team", "Pos", "Year", "Pts Won", "Pts Max", "Share"]
predictors = [c for c in stats.columns if c not in not_features]
dev = stats[stats["Year"] <= 2024]

rng = np.random.default_rng(SEED)


def sample_params():
    return {
        "n_estimators": int(rng.integers(200, 1200)),
        "learning_rate": float(10 ** rng.uniform(-2.2, -1.0)),
        "num_leaves": int(rng.integers(8, 64)),
        "min_child_samples": int(rng.integers(10, 80)),
        "subsample": float(rng.uniform(0.5, 1.0)),
        "subsample_freq": 1,
        "colsample_bytree": float(rng.uniform(0.5, 1.0)),
    }


def score(params):
    mses, hits = [], []
    for year in VALIDATION_SEASONS:
        train, val = dev[dev["Year"] < year], dev[dev["Year"] == year]
        model = lgb.LGBMRegressor(**params, random_state=SEED, verbose=-1)
        model.fit(train[predictors], train[target])
        pred = model.predict(val[predictors])
        mses.append(mean_squared_error(val[target], pred))
        hits.append(val[target].iloc[int(np.argmax(pred))] == val[target].max())
    return float(np.mean(mses)), float(np.mean(hits))


results = []
for i in range(N_TRIALS):
    params = sample_params()
    mse, hit = score(params)
    results.append({"params": params, "cv_mse": mse, "mvp_hit_rate": hit})
    print(f"trial {i + 1:02d}/{N_TRIALS}  mse={mse:.5f}  mvp_hit_rate={hit:.2f}")

best = min(results, key=lambda r: r["cv_mse"])
print("\nBest:", json.dumps(best, indent=2))
with open("best_params.json", "w") as f:
    json.dump({"params": best["params"], "cv_mse": best["cv_mse"],
               "mvp_hit_rate": best["mvp_hit_rate"],
               "validation_seasons": [min(VALIDATION_SEASONS), max(VALIDATION_SEASONS)]}, f, indent=2)
