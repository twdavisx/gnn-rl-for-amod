import pandas as pd

# Match main.py: 3-minute bins and the 7--10 a.m. demand window.
BIN_MINUTES = 3
START_MINUTE = 7 * 60
END_MINUTE = 10 * 60


def main():

    daily = pd.read_csv("manhattan_daily_counts.csv", parse_dates=["day"])

    required = {"day", "time_stamp", "origin", "destination", "count"}
    missing = required - set(daily.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # start timestep at 0
    daily = daily[(daily.time_stamp >= START_MINUTE) & (daily.time_stamp < END_MINUTE)].copy()
    daily["time_bin"] = (daily.time_stamp - START_MINUTE) // BIN_MINUTES


    #Pearson dispersion statistic calculation 
    counts = (daily.groupby(["day", "origin", "destination", "time_bin"], as_index=False)
                   ["count"].sum())
    n_days = counts["day"].nunique()

    cells = (counts.assign(count_squared=counts["count"] ** 2)
             .groupby(["origin", "destination", "time_bin"], as_index=False)
             .agg(total=("count", "sum"), sum_squares=("count_squared", "sum")))
    cells["lambda"] = cells["total"] / n_days
    pearson_chi2 = (cells["sum_squares"] / cells["lambda"] - cells["total"]).sum()
    n_cells = len(cells)
    phi = pearson_chi2 / (n_cells * (n_days - 1))

    print(f"Days: {n_days}")
    print(f"OD/time-bin cells: {n_cells}")
    print(f"Pearson dispersion (phi): {phi:.4f}")
    print("Near 1: Poisson-like; above 1: overdispersed; below 1: underdispersed.")


if __name__ == "__main__":
    main()
