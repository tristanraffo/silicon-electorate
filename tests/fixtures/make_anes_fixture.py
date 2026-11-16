"""Generate the synthetic ANES-shaped CSV used by the tests (SPEC §13 M5).

    uv run python tests/fixtures/make_anes_fixture.py

Nothing here is ANES data. The column names are fake on purpose (SYN_*), and the file
imitates the *shape* of an ANES release: numeric codes, negative missing-data codes,
998/999 on thermometers, a weight column, and unrelated extra columns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).with_name("anes_synthetic.csv")
N = 400
# five fictional candidates on a left-right line
POSITIONS = {"SYN_FT_A": -1.5, "SYN_FT_B": -0.5, "SYN_FT_C": 0.0, "SYN_FT_D": 0.7, "SYN_FT_E": 1.6}


def make(seed: int = 2026) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ideology = rng.normal(size=N)  # latent left-right position
    df = pd.DataFrame({"SYN_ID": np.arange(100001, 100001 + N)})
    df["SYN_WT"] = np.round(rng.gamma(4.0, 0.25, size=N), 4)
    df.loc[rng.choice(N, 6, replace=False), "SYN_WT"] = 0.0  # not in the weighted sample
    df.loc[rng.choice(N, 4, replace=False), "SYN_WT"] = np.nan

    for col, pos in POSITIONS.items():
        warmth = 85 - 28 * np.abs(ideology - pos) + rng.normal(0, 10, size=N)
        values = np.clip(np.round(warmth / 5) * 5, 0, 100)  # people answer in fives
        missing = rng.random(N)
        values[missing < 0.04] = -9  # refused
        values[(missing >= 0.04) & (missing < 0.07)] = 998  # don't know
        values[(missing >= 0.07) & (missing < 0.08)] = 999  # don't recognise
        df[col] = values.astype(int)
    df.loc[:11, ["SYN_FT_A", "SYN_FT_B", "SYN_FT_C"]] = -9  # 12 people rate at most 2

    age = rng.integers(18, 81, size=N)
    age[rng.random(N) < 0.03] = -9
    df["SYN_AGE"] = age
    df["SYN_GENDER"] = rng.choice([1, 2, -9], size=N, p=[0.48, 0.5, 0.02])
    df["SYN_RACE"] = rng.choice(
        [1, 2, 3, 4, 5, 6, -8], size=N, p=[0.6, 0.12, 0.15, 0.05, 0.02, 0.04, 0.02]
    )
    df["SYN_EDUC"] = rng.choice([1, 2, 3, 4, 5, -9], size=N, p=[0.08, 0.27, 0.3, 0.2, 0.13, 0.02])
    df["SYN_INCOME"] = rng.choice(
        [1, 2, 3, 4, 5, -9, -5], size=N, p=[0.2, 0.2, 0.2, 0.17, 0.13, 0.06, 0.04]
    )
    df["SYN_STATE"] = rng.choice(["Ohio", "Texas", "Oregon", "Georgia", "Maine", ""], size=N)
    pid = np.clip(np.round(4 + 1.6 * ideology + rng.normal(0, 0.8, size=N)), 1, 7).astype(int)
    pid[rng.random(N) < 0.03] = -8
    df["SYN_PID7"] = pid
    ideo = np.clip(np.round(4 + 1.4 * ideology + rng.normal(0, 0.9, size=N)), 1, 7).astype(int)
    ideo[rng.random(N) < 0.15] = 99  # "haven't thought much about this"
    df["SYN_IDEO7"] = ideo
    df["SYN_RELIG"] = rng.choice([1, 2, 3, 4, 5, -9], size=N, p=[0.2, 0.1, 0.15, 0.2, 0.33, 0.02])
    df["SYN_VOTE"] = rng.choice([1, 2, 3, 4, -1], size=N, p=[0.33, 0.3, 0.05, 0.27, 0.05])
    df["SYN_UNRELATED_1"] = rng.integers(0, 5, size=N)
    df["SYN_UNRELATED_2"] = rng.normal(size=N).round(3)
    return df


if __name__ == "__main__":
    make().to_csv(OUT, index=False)
    print(f"wrote {OUT} ({N} rows)")
