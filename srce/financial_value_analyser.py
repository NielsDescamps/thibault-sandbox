"""Decompose a BESS dispatch scenario's energy flows into who creates which
value, and make explicit which party -- the site (client) or the BESS
operator -- that value accrues to *by default* under the current billing
convention (DA price + grid fees, applied only to what crosses the grid
meter; see `Scenario_Comparison.ipynb`'s `scenario_metrics`).

Two fundamentally different kinds of number appear here, and this module
keeps them visibly separate rather than blurring them into one bill:

- `grid_charge`/`grid_discharge` are real cash flows -- they're the portion
  of this scenario's `charge_mwh`/`discharge_mwh` that actually crossed the
  grid meter (i.e. they're already inside `grid_offtake_mwh`/
  `grid_injection_mwh`, and thus already inside `energy_cost`/`fee_cost`).
  They belong to whoever runs the battery's arbitrage strategy.
- `sc_charge`/`sc_discharge` (self-consumption) never cross the grid meter
  at all -- they're not cash in this scenario's bill. They're *opportunity*
  values: what that energy would have been worth had it crossed the meter
  instead. For `sc_discharge` (covering the site's own load) that
  counterfactual is unambiguous -- it's a direct avoided import, and the
  site's own meter shows it. For `sc_charge` (absorbing the site's own
  excess generation) there are genuinely two different valid counterfactuals
  depending on who would otherwise have captured the export, and this module
  reports both rather than picking one.
"""

import numpy as np
import pandas as pd

FLOAT_TOL = 1e-6

AMBIGUOUS_ACCRUAL_NOTE = (
    "Ambiguous / defaults to BESS operator today -- depends on real-time "
    "netting vs. forecast/formula settlement between the two BRPs; not yet "
    "confirmed for the actual site contracts."
)


def decompose_dispatch_flows(df, result):
    """Split a scenario's charge/discharge into self-consumption vs. grid flows.

    Uses the *net* residual position (`inj_mwh - off_mwh`), not the raw
    `off_mwh`/`inj_mwh` columns separately, since some sites have both
    nonzero in the same interval (multi-zone metering) -- the same
    convention already used by `compute_self_consumption_schedule`.

    Parameters
    ----------
    df : pd.DataFrame
        Must have `off_mwh`, `inj_mwh` columns.
    result : dict
        A scenario result dict with `charge_mwh`, `discharge_mwh`
        (list/array, one value per interval).

    Returns
    -------
    dict[str, np.ndarray]
        `r_plus`, `r_minus` (net residual injection/offtake, for reference),
        `sc_charge`, `grid_charge`, `sc_discharge`, `grid_discharge` -- one
        value per interval, in MWh.
    """
    off_mwh = df["off_mwh"].to_numpy()
    inj_mwh = df["inj_mwh"].to_numpy()
    charge_mwh = np.asarray(result["charge_mwh"], dtype=float)
    discharge_mwh = np.asarray(result["discharge_mwh"], dtype=float)

    r_plus = np.maximum(inj_mwh - off_mwh, 0.0)
    r_minus = np.maximum(off_mwh - inj_mwh, 0.0)

    sc_charge = np.minimum(charge_mwh, r_plus)
    grid_charge = charge_mwh - sc_charge
    sc_discharge = np.minimum(discharge_mwh, r_minus)
    grid_discharge = discharge_mwh - sc_discharge

    # should always hold by construction -- if it doesn't, something upstream
    # (e.g. a scenario's result dict) is inconsistent
    assert np.allclose(sc_charge + grid_charge, charge_mwh, atol=FLOAT_TOL), (
        "sc_charge + grid_charge should always equal charge_mwh by construction"
    )
    assert np.allclose(sc_discharge + grid_discharge, discharge_mwh, atol=FLOAT_TOL), (
        "sc_discharge + grid_discharge should always equal discharge_mwh by construction"
    )

    return {
        "r_plus": r_plus,
        "r_minus": r_minus,
        "sc_charge": sc_charge,
        "grid_charge": grid_charge,
        "sc_discharge": sc_discharge,
        "grid_discharge": grid_discharge,
    }


def _volume_weighted_price(volume, price):
    """Returns (avg_price_or_None, raw_energy_value). raw_energy_value is
    exactly 0.0 when volume totals ~0, so callers never have to special-case
    multiplying by a None average price."""
    total = float(volume.sum())
    raw_energy = float((volume * price).sum())
    if abs(total) < FLOAT_TOL:
        return None, 0.0
    return raw_energy / total, raw_energy


def grid_volume_and_price(df, result):
    """Volume and DA volume-weighted average price of what actually crossed
    the grid connection in this scenario.

    Deliberately doesn't compute cost/fees -- that's `scenario_metrics`'s
    job (in the notebook); this is just the underlying volumes and prices
    those cost numbers are built from, made explicit.
    """
    price = df["price"].to_numpy()
    grid_offtake = np.asarray(result["grid_offtake_mwh"], dtype=float)
    grid_injection = np.asarray(result["grid_injection_mwh"], dtype=float)

    off_avg_price, _ = _volume_weighted_price(grid_offtake, price)
    inj_avg_price, _ = _volume_weighted_price(grid_injection, price)

    return {
        "grid_offtake_mwh": float(grid_offtake.sum()),
        "grid_offtake_avg_price_eur_mwh": off_avg_price,
        "grid_injection_mwh": float(grid_injection.sum()),
        "grid_injection_avg_price_eur_mwh": inj_avg_price,
    }


def financial_value_table(df, result, params):
    """Decompose a scenario's dispatch into flow buckets and value each one.

    One row per bucket: Volume (MWh), the DA volume-weighted average price
    for that bucket's own volume (`n/a`/`None` if the bucket's total volume
    is ~0), Energy value, Grid-fee value, Total value (= Energy + Grid-fee),
    and who that value accrues to today. See the module docstring for why
    the self-consumption rows are opportunity values, not bill line items,
    and why the injection-leg (`sc_charge`) bucket gets two rows instead of
    one -- it's shown as two competing counterfactuals on purpose, not
    collapsed into a single number.

    Parameters
    ----------
    df : pd.DataFrame
        Must have `off_mwh`, `inj_mwh`, `price` columns.
    result : dict
        Scenario result dict (`charge_mwh`, `discharge_mwh`, `grid_offtake_mwh`,
        `grid_injection_mwh`).
    params : dict
        Must have `GRID_FEE_OFFTAKE_EUR_MWH`, `GRID_FEE_INJECTION_EUR_MWH`.

    Returns
    -------
    pd.DataFrame
        Columns: Flow, Volume (MWh), Avg DA price (EUR/MWh), Energy value
        (EUR), Grid-fee value (EUR), Total value (EUR), Accrues to.
    """
    price = df["price"].to_numpy()
    flows = decompose_dispatch_flows(df, result)
    fee_offtake = params["GRID_FEE_OFFTAKE_EUR_MWH"]
    fee_injection = params["GRID_FEE_INJECTION_EUR_MWH"]

    rows = []

    def add_row(name, volume_series, fee_rate, sign, accrues_to):
        volume = float(volume_series.sum())
        avg_price, raw_energy = _volume_weighted_price(volume_series, price)
        energy_value = sign * raw_energy
        fee_value = sign * volume * fee_rate
        rows.append({
            "Flow": name,
            "Volume (MWh)": volume,
            "Avg DA price (EUR/MWh)": avg_price,
            "Energy value (EUR)": energy_value,
            "Grid-fee value (EUR)": fee_value,
            "Total value (EUR)": energy_value + fee_value,
            "Accrues to": accrues_to,
        })

    # --- real cash flows: already inside this scenario's own energy_cost/fee_cost ---
    add_row(
        "Grid charging (arbitrage cost)",
        flows["grid_charge"], fee_offtake, -1.0,
        "BESS operator",
    )
    add_row(
        "Grid discharging (arbitrage revenue)",
        flows["grid_discharge"], fee_injection, 1.0,
        "BESS operator",
    )

    # --- counterfactual / opportunity values: never cross the grid meter ---
    add_row(
        "Self-consumption discharge -- avoided import (offtake leg)",
        flows["sc_discharge"], fee_offtake, 1.0,
        "Site (client)",
    )
    add_row(
        "Self-consumption charge -- avoided export (injection leg), "
        "site-side: if this had been exported by the site",
        flows["sc_charge"], fee_injection, 1.0,
        AMBIGUOUS_ACCRUAL_NOTE,
    )
    add_row(
        "Self-consumption charge -- avoided export (injection leg), "
        "BESS-side: if the operator had instead charged this from the grid",
        flows["sc_charge"], fee_offtake, 1.0,
        AMBIGUOUS_ACCRUAL_NOTE,
    )

    return pd.DataFrame(rows)


def compare_leg_volume_estimates(rough_offtake_leg_mwh, rough_injection_leg_mwh, flows, tolerance=0.05):
    """Compare `scenario_metrics`'s rough offtake/injection-leg volumes
    (computed on the raw `off_mwh`/`inj_mwh` columns directly, not net of
    multi-zone metering) against this module's net-residual-based
    `sc_discharge`/`sc_charge` totals.

    These two are genuinely different calculations and are expected to be
    close but not necessarily identical. Flags (returns `flagged: True`,
    doesn't raise) when the gap exceeds `tolerance` of the larger volume.

    Parameters
    ----------
    rough_offtake_leg_mwh, rough_injection_leg_mwh : float
        `scenario_metrics`'s `offtake_leg_mwh`/`injection_leg_mwh`.
    flows : dict
        Output of `decompose_dispatch_flows`.
    tolerance : float
        Fraction of the larger volume beyond which the gap is flagged.

    Returns
    -------
    dict
        `offtake_leg` and `injection_leg`, each with `rough_mwh`,
        `precise_mwh`, `gap_mwh`, `gap_pct_of_larger`, `flagged`.
    """
    sc_discharge_mwh = float(flows["sc_discharge"].sum())
    sc_charge_mwh = float(flows["sc_charge"].sum())

    def _compare(rough, precise):
        larger = max(abs(rough), abs(precise), FLOAT_TOL)
        gap = abs(rough - precise)
        return {
            "rough_mwh": rough,
            "precise_mwh": precise,
            "gap_mwh": gap,
            "gap_pct_of_larger": gap / larger,
            "flagged": gap / larger > tolerance,
        }

    return {
        "offtake_leg": _compare(rough_offtake_leg_mwh, sc_discharge_mwh),
        "injection_leg": _compare(rough_injection_leg_mwh, sc_charge_mwh),
    }
