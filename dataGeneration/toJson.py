import json
import pandas as pd

matrix = pd.read_csv("manhattan_matrix.csv")

# number of regions (zone ids are 0..N-1)
N = int(max(matrix["origin"].max(), matrix["destination"].max())) + 1

# ---- fleet (totalAcc) ----
# match nyc4x4's fleet/demand ratio (~0.0718) so the default regime is comparably loose
total_demand = matrix["demand"].sum()
fleet = 0.0718 * total_demand
# env reads the row where hour == json_hr + round(json_tstep/2*tf/60); emit all 24 hrs to be safe
totalAcc = [{"hour": h, "acc": float(fleet)} for h in range(24)]

# ---- rebTime (all N*N pairs, at each hour) ----
# reb_time ~= observed avg trip time for that OD; fill unobserved pairs with a fallback
reb = matrix.groupby(["origin", "destination"])["travel_time"].mean()
reb_obs = {(int(o), int(d)): float(v) for (o, d), v in reb.items()}
print(len(reb_obs), "of", N * N, "OD pairs observed")
global_mean = float(reb.mean())

reb_time_by_pair = {}
for o in range(N):
    for d in range(N):
        if (o, d) in reb_obs:
            rt = reb_obs[(o, d)]          # observed
        elif (d, o) in reb_obs:
            rt = reb_obs[(d, o)]          # reverse-direction fallback
        elif o == d:
            rt = 1.0                      # self-loop: minimal reposition time (min 1 timestep in env)
        else:
            rt = global_mean              # last-resort fallback (swap for centroid distance if many missing)
        reb_time_by_pair[(o, d)] = rt

rebTime = [
    {"time_stamp": h, "origin": int(o), "destination": int(d), "reb_time": float(rt)}
    for (o, d), rt in reb_time_by_pair.items()
    for h in range(24)
]

# ---- demand (one row per observed (minute, origin, destination)) ----
demand = [
    {"time_stamp": int(r.time_stamp), "origin": int(r.origin), "destination": int(r.destination),
     "demand": float(r.demand), "travel_time": float(r.travel_time), "price": float(r.price)}
    for r in matrix.itertuples(index=False)
]

# ---- assemble + write ----
scenario = {
    "nlat": N,          # both read by the env; nlat*nlon need not equal N when "region" is present
    "nlon": 1,
    "region": N,
    "demand": demand,
    "rebTime": rebTime,
    "totalAcc": totalAcc,
}

out_path = "scenario_nycReal.json"
with open(out_path, "w") as f:
    json.dump(scenario, f)