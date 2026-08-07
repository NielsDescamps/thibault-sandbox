from ortools.linear_solver import pywraplp

REQUIRED_PARAMS = (
    "CAPACITY_MWH",
    "INITIAL_SOC_MWH",
    "MAX_CHARGE_MWH",
    "MAX_DISCHARGE_MWH",
    "CHARGE_EFFICIENCY",
    "DISCHARGE_EFFICIENCY",
    "MAX_OFFTAKE_MWH",
    "MAX_INJECTION_MWH",
    "GRID_FEE_OFFTAKE_EUR_MWH",
    "GRID_FEE_INJECTION_EUR_MWH",
    "MAX_CYCLES_PER_DAY",
)


def solve_battery_dispatch(df, params, ppa_charge_target=None, ppa_discharge_target=None):
    """Solve the BESS dispatch MILP: minimize DA-price + grid-fee cost.

    Mutual-exclusivity binaries prevent the solver from charging and
    discharging (or drawing from and injecting into the grid) in the same
    interval, and a daily cap limits equivalent full charge cycles per
    calendar day. Optionally takes a mandatory self-consumption floor as
    per-interval lower bounds on `charge`/`discharge` -- since setting every
    extra-arbitrage variable to zero always satisfies those bounds, adding a
    floor can never make the MILP infeasible on its own.

    Parameters
    ----------
    df : pd.DataFrame
        Must have `off_mwh`, `inj_mwh`, `price`, `dates` columns, sorted by
        `dates` with a clean 0..n-1 index (`reset_index(drop=True)`).
    params : dict
        CAPACITY_MWH, INITIAL_SOC_MWH, MAX_CHARGE_MWH, MAX_DISCHARGE_MWH,
        CHARGE_EFFICIENCY, DISCHARGE_EFFICIENCY, MAX_OFFTAKE_MWH,
        MAX_INJECTION_MWH, GRID_FEE_OFFTAKE_EUR_MWH,
        GRID_FEE_INJECTION_EUR_MWH, MAX_CYCLES_PER_DAY.
    ppa_charge_target, ppa_discharge_target : list of float, optional
        Per-interval lower bounds on charge/discharge (the mandatory
        self-consumption floor, e.g. from `compute_self_consumption_schedule`).
        Omit for unconstrained arbitrage.

    Returns
    -------
    dict
        `charge_mwh`, `discharge_mwh`, `soc_mwh`, `grid_offtake_mwh`,
        `grid_injection_mwh` -- each a plain list of floats, one per interval.
        Only plain Python values are returned: OR-Tools' underlying solver
        objects are not safe to keep alive across repeated solves in one
        process, so nothing from `pywraplp` escapes this function.
    """
    missing = [p for p in REQUIRED_PARAMS if p not in params]
    if missing:
        raise ValueError(f"params is missing required key(s): {missing}")

    CAPACITY_MWH = params["CAPACITY_MWH"]
    INITIAL_SOC_MWH = params["INITIAL_SOC_MWH"]
    MAX_CHARGE_MWH = params["MAX_CHARGE_MWH"]
    MAX_DISCHARGE_MWH = params["MAX_DISCHARGE_MWH"]
    CHARGE_EFFICIENCY = params["CHARGE_EFFICIENCY"]
    DISCHARGE_EFFICIENCY = params["DISCHARGE_EFFICIENCY"]
    MAX_OFFTAKE_MWH = params["MAX_OFFTAKE_MWH"]
    MAX_INJECTION_MWH = params["MAX_INJECTION_MWH"]
    GRID_FEE_OFFTAKE_EUR_MWH = params["GRID_FEE_OFFTAKE_EUR_MWH"]
    GRID_FEE_INJECTION_EUR_MWH = params["GRID_FEE_INJECTION_EUR_MWH"]
    MAX_CYCLES_PER_DAY = params["MAX_CYCLES_PER_DAY"]

    n = len(df)
    solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        raise RuntimeError("Could not create the CBC MILP solver -- check the OR-Tools install.")

    charge = [solver.NumVar(0, MAX_CHARGE_MWH, f"charge_{t}") for t in range(n)]
    discharge = [solver.NumVar(0, MAX_DISCHARGE_MWH, f"discharge_{t}") for t in range(n)]
    soc = [solver.NumVar(0, CAPACITY_MWH, f"soc_{t}") for t in range(n)]
    grid_offtake = [solver.NumVar(0, MAX_OFFTAKE_MWH, f"grid_offtake_{t}") for t in range(n)]
    grid_injection = [solver.NumVar(0, MAX_INJECTION_MWH, f"grid_injection_{t}") for t in range(n)]
    is_charging = [solver.BoolVar(f"is_charging_{t}") for t in range(n)]
    is_grid_offtake = [solver.BoolVar(f"is_grid_offtake_{t}") for t in range(n)]

    for t in range(n):
        prev_soc = INITIAL_SOC_MWH if t == 0 else soc[t - 1]
        solver.Add(soc[t] == prev_soc + charge[t] * CHARGE_EFFICIENCY - discharge[t] / DISCHARGE_EFFICIENCY)

        # mutual exclusivity: charge or discharge, never both
        solver.Add(charge[t] <= MAX_CHARGE_MWH * is_charging[t])
        solver.Add(discharge[t] <= MAX_DISCHARGE_MWH * (1 - is_charging[t]))

        # net exchange at the grid connection point: load/generation (off, inj) have
        # priority -- they're fixed inputs the battery cannot change. The battery
        # can only adjust charge[t]/discharge[t] to keep the *combined* result
        # within the connection limits.
        net_grid_t = df["off_mwh"].iat[t] - df["inj_mwh"].iat[t] + charge[t] - discharge[t]
        solver.Add(grid_offtake[t] - grid_injection[t] == net_grid_t)

        # mutual exclusivity: draw from the grid or inject into it, never both
        solver.Add(grid_offtake[t] <= MAX_OFFTAKE_MWH * is_grid_offtake[t])
        solver.Add(grid_injection[t] <= MAX_INJECTION_MWH * (1 - is_grid_offtake[t]))

        if ppa_charge_target is not None:
            solver.Add(charge[t] >= ppa_charge_target[t])
            solver.Add(discharge[t] >= ppa_discharge_target[t])

    # cycle limit: an "equivalent full cycle" is CAPACITY_MWH worth of charging
    # throughput. Capping total charge[t] per calendar day at
    # MAX_CYCLES_PER_DAY * CAPACITY_MWH limits the battery to that many full
    # charge cycles per day, without needing extra binaries to count discrete
    # charge/discharge events.
    for _day, idx in df.groupby(df["dates"].dt.date).indices.items():
        solver.Add(solver.Sum([charge[t] for t in idx]) <= MAX_CYCLES_PER_DAY * CAPACITY_MWH)

    # cost[t] = price[t] * (off[t] - inj[t] + charge[t] - discharge[t]) + grid fees
    objective = solver.Objective()
    for t in range(n):
        price = df["price"].iat[t]
        objective.SetCoefficient(charge[t], price)
        objective.SetCoefficient(discharge[t], -price)
        objective.SetCoefficient(grid_offtake[t], GRID_FEE_OFFTAKE_EUR_MWH)
        objective.SetCoefficient(grid_injection[t], -GRID_FEE_INJECTION_EUR_MWH)
    objective.SetMinimization()

    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError(
            "Solver did not find an optimal solution -- if you tightened the grid "
            "connection limits, check that off/inj alone (without the battery) "
            "never exceed them, since load/generation are fixed and cannot be curtailed. "
            f"(status code: {status})"
        )

    result = {
        "charge_mwh": [v.solution_value() for v in charge],
        "discharge_mwh": [v.solution_value() for v in discharge],
        "soc_mwh": [v.solution_value() for v in soc],
        "grid_offtake_mwh": [v.solution_value() for v in grid_offtake],
        "grid_injection_mwh": [v.solution_value() for v in grid_injection],
    }
    del solver, charge, discharge, soc, grid_offtake, grid_injection, is_charging, is_grid_offtake, objective
    return result


def compute_self_consumption_schedule(df, params):
    """Deterministic forward simulation of a "pure self-consumption" battery.

    Not an optimization -- there's no decision freedom. At each interval,
    based on the *net* position (`off_mwh - inj_mwh`, not the raw columns
    separately -- some sites have both nonzero in the same interval due to
    multi-zone metering, and the net is what's physically consistent with a
    single grid connection), the battery absorbs as much of a net exporter
    interval's excess as its power limit and *current* capacity headroom
    allow, or releases as much of a net importer interval's deficit as its
    power limit and *currently available* stored energy allow.

    Parameters
    ----------
    df : pd.DataFrame
        Must have `off_mwh`, `inj_mwh` columns, sorted by date with a clean
        0..n-1 index.
    params : dict
        CAPACITY_MWH, INITIAL_SOC_MWH, MAX_CHARGE_MWH, MAX_DISCHARGE_MWH,
        CHARGE_EFFICIENCY, DISCHARGE_EFFICIENCY. (The grid-side params in
        `REQUIRED_PARAMS` aren't needed here -- this scenario never touches
        the grid beyond whatever's left over after self-consumption.)

    Returns
    -------
    dict
        Same shape as `solve_battery_dispatch`'s result (`charge_mwh`,
        `discharge_mwh`, `soc_mwh`, `grid_offtake_mwh`, `grid_injection_mwh`),
        so it can be used as a scenario result directly. `charge_mwh` and
        `discharge_mwh` here are also exactly the mandatory floor to pass as
        `ppa_charge_target`/`ppa_discharge_target` to `solve_battery_dispatch`
        for a "self-consumption first, arbitrage with what's left" scenario.
    """
    CAPACITY_MWH = params["CAPACITY_MWH"]
    INITIAL_SOC_MWH = params["INITIAL_SOC_MWH"]
    MAX_CHARGE_MWH = params["MAX_CHARGE_MWH"]
    MAX_DISCHARGE_MWH = params["MAX_DISCHARGE_MWH"]
    CHARGE_EFFICIENCY = params["CHARGE_EFFICIENCY"]
    DISCHARGE_EFFICIENCY = params["DISCHARGE_EFFICIENCY"]

    n = len(df)
    charge_target = [0.0] * n
    discharge_target = [0.0] * n
    soc_trace = [0.0] * n
    prev_soc = INITIAL_SOC_MWH
    for t in range(n):
        net = df["off_mwh"].iat[t] - df["inj_mwh"].iat[t]
        if net < 0:  # net exporter -> must absorb
            headroom = max(CAPACITY_MWH - prev_soc, 0) / CHARGE_EFFICIENCY
            c, d = min(-net, MAX_CHARGE_MWH, headroom), 0.0
        elif net > 0:  # net importer -> must release
            available = max(prev_soc, 0) * DISCHARGE_EFFICIENCY
            c, d = 0.0, min(net, MAX_DISCHARGE_MWH, available)
        else:
            c, d = 0.0, 0.0
        charge_target[t] = c
        discharge_target[t] = d
        prev_soc = prev_soc + c * CHARGE_EFFICIENCY - d / DISCHARGE_EFFICIENCY
        soc_trace[t] = prev_soc

    net_grid = [
        df["off_mwh"].iat[t] - df["inj_mwh"].iat[t] + charge_target[t] - discharge_target[t]
        for t in range(n)
    ]
    return {
        "charge_mwh": charge_target,
        "discharge_mwh": discharge_target,
        "soc_mwh": soc_trace,
        "grid_offtake_mwh": [max(v, 0.0) for v in net_grid],
        "grid_injection_mwh": [max(-v, 0.0) for v in net_grid],
    }
