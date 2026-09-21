"""
Transient rainfall infiltration -> pore pressure head.

Two solvers, deliberately:

1. `pressure_head_iverson` -- the linearised analytic solution of Iverson
   (2000) as implemented in TRIGRS (Baum, Savage & Godt 2008), with
   superposition over an arbitrary rainfall hyetograph. This is the
   workhorse: it is vectorised over Monte Carlo draws and runs fast enough
   to build a training corpus of hundreds of thousands of scenarios.

2. `pressure_head_numerical` -- a Crank-Nicolson finite-difference solution
   of the same linear diffusion problem, solved for the PERTURBATION from
   the initial steady state so that it uses exactly the same boundary
   condition as the analytic form. Slower, and used to VERIFY the analytic
   solver rather than in production.

Having both matters. "We verified our analytic solution against an
independent numerical solver, and they agree to within a few millimetres of
pressure head" is a sentence that survives scrutiny; a single unverified
closed form is not.

Formulation
-----------
The linearised problem is

    d(psi')/dt = D1 * d2(psi')/dZ2,     D1 = D0 / cos^2(beta)

for the perturbation psi' about the steady profile

    psi_steady(Z) = (Z - d) * (cos^2(beta) - Iz_LT/Ks)

with a surface flux condition d(psi')/dZ|_{Z=0} = -(Iz/Ks) and psi' -> 0 at
depth. The analytic response to a step of flux Iz switched on at t=0 is

    psi'(Z,t) = (Iz/Ks) * 2 * sqrt(D1*t) * ierfc( Z / (2*sqrt(D1*t)) )

and an arbitrary hyetograph follows by superposing switch-on and switch-off
steps at each interval boundary.

Sign convention: pressure head psi is positive when the soil is above
atmospheric pressure (saturated, destabilising) and negative in suction.
Depth Z is measured vertically downward from the ground surface, in metres.

References
----------
Iverson, R.M. (2000). Landslide triggering by rain infiltration.
    Water Resources Research 36(7), 1897-1910.
Baum, R.L., Savage, W.Z. & Godt, J.W. (2008). TRIGRS -- A FORTRAN program
    for transient rainfall infiltration and grid-based regional
    slope-stability analysis, version 2.0. USGS Open-File Report 2008-1159.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded
from scipy.special import erfc


def ierfc(eta: np.ndarray) -> np.ndarray:
    """Integral of the complementary error function.

    ierfc(x) = exp(-x^2)/sqrt(pi) - x * erfc(x)

    Underflows gracefully to 0 for large x, which is the correct limit.
    """
    eta = np.asarray(eta, dtype=float)
    return np.exp(-np.square(eta)) / np.sqrt(np.pi) - eta * erfc(eta)


def steady_profile(
    depths_m: np.ndarray,
    water_table_depth_m: np.ndarray | float,
    slope_rad: float,
    steady_ratio: float,
) -> np.ndarray:
    """Initial steady-state pressure head profile.

    psi_steady(Z) = (Z - d) * (cos^2(beta) - Iz_LT/Ks)

    `steady_ratio` is the long-term (antecedent) infiltration as a fraction
    of Ksat, so raising it wets the profile.
    """
    depths = np.atleast_1d(np.asarray(depths_m, dtype=float))
    wt = np.atleast_1d(np.asarray(water_table_depth_m, dtype=float))
    beta_term = np.cos(slope_rad) ** 2 - steady_ratio
    return (depths[None, :] - wt[:, None]) * beta_term


def pressure_head_iverson(
    depths_m: np.ndarray,
    time_edges_s: np.ndarray,
    intensity_ms: np.ndarray,
    *,
    ksat_ms: np.ndarray | float,
    diffusivity_m2s: np.ndarray | float,
    slope_rad: float,
    water_table_depth_m: np.ndarray | float,
    steady_ratio: float = 0.1,
) -> np.ndarray:
    """Pressure head field psi(Z, t) under a rainfall hyetograph.

    Parameters
    ----------
    depths_m : (n_z,)
        Vertical depths at which to evaluate, metres below surface.
    time_edges_s : (n_t + 1,)
        Boundaries of the rainfall intervals, seconds. Output is returned at
        the interval *ends*, i.e. at time_edges_s[1:].
    intensity_ms : (n_t,)
        Surface water flux during each interval, m/s. Convert from mm/h with
        `mm_per_hour / 1000.0 / 3600.0`.
    ksat_ms, diffusivity_m2s, water_table_depth_m : scalar or (n_draw,)
        Per-draw soil parameters.
    slope_rad : float
        Slope angle, radians.
    steady_ratio : float
        Antecedent infiltration as a fraction of Ksat.

    Returns
    -------
    psi : (n_draw, n_t, n_z), or (n_t, n_z) if every parameter is scalar.

    Notes
    -----
    Surface flux is capped at Ksat -- water that cannot infiltrate runs off
    rather than driving pressure head. Without this cap an intense short
    burst produces physically absurd pressures.
    """
    depths = np.atleast_1d(np.asarray(depths_m, dtype=float))
    edges = np.asarray(time_edges_s, dtype=float)
    intensity = np.asarray(intensity_ms, dtype=float)

    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("time_edges_s must be 1-D with at least two entries")
    if intensity.shape[0] != edges.size - 1:
        raise ValueError(
            f"intensity_ms has {intensity.shape[0]} intervals but "
            f"time_edges_s implies {edges.size - 1}"
        )
    if np.any(np.diff(edges) <= 0):
        raise ValueError("time_edges_s must be strictly increasing")

    scalar_params = (
        np.ndim(ksat_ms) == 0
        and np.ndim(diffusivity_m2s) == 0
        and np.ndim(water_table_depth_m) == 0
    )

    ks = np.atleast_1d(np.asarray(ksat_ms, dtype=float))
    d0 = np.atleast_1d(np.asarray(diffusivity_m2s, dtype=float))
    wt = np.atleast_1d(np.asarray(water_table_depth_m, dtype=float))

    n_draw = max(ks.size, d0.size, wt.size)
    ks = np.broadcast_to(ks, (n_draw,))
    d0 = np.broadcast_to(d0, (n_draw,))
    wt = np.broadcast_to(wt, (n_draw,))

    cos2b = np.cos(slope_rad) ** 2
    d1 = d0 / cos2b

    # Surface flux cannot exceed Ksat; the excess runs off. (n_draw, n_t)
    flux_ratio = np.minimum(intensity[None, :] / ks[:, None], 1.0)

    out_times = edges[1:]
    n_t = out_times.size
    n_z = depths.size

    psi_steady = steady_profile(depths, wt, slope_rad, steady_ratio)
    psi = np.repeat(psi_steady[:, None, :], n_t, axis=1)

    sqrt_d1 = np.sqrt(d1)[:, None]  # (n_draw, 1)

    for n in range(n_t):
        ratio_n = flux_ratio[:, n]
        if not np.any(ratio_n > 0.0):
            continue
        ratio_n = ratio_n[:, None, None]  # (n_draw, 1, 1)

        for t_edge, sign in ((edges[n], 1.0), (edges[n + 1], -1.0)):
            dt = out_times - t_edge            # (n_t,)
            active = dt > 0.0
            if not np.any(active):
                continue
            dt_safe = np.where(active, dt, 1.0)[None, :]         # (1, n_t)
            root = sqrt_d1 * np.sqrt(dt_safe)                    # (n_draw, n_t)
            eta = depths[None, None, :] / (2.0 * root[:, :, None])
            term = 2.0 * root[:, :, None] * ierfc(eta)
            term = np.where(active[None, :, None], term, 0.0)
            psi += sign * ratio_n * term

    # Physical ceiling: no artesian pressure in a hillslope veneer.
    psi = np.minimum(psi, depths[None, None, :])

    if scalar_params and n_draw == 1:
        return psi[0]
    return psi


def pressure_head_numerical(
    depths_m: np.ndarray,
    time_edges_s: np.ndarray,
    intensity_ms: np.ndarray,
    *,
    ksat_ms: float,
    diffusivity_m2s: float,
    slope_rad: float,
    water_table_depth_m: float,
    steady_ratio: float = 0.1,
    n_sub: int = 20,
    n_grid: int = 240,
    far_field_factor: float = 6.0,
) -> np.ndarray:
    """Crank-Nicolson solution of the same problem, for verification.

    Solves for the perturbation psi' about the steady profile, with the same
    surface flux boundary condition as the analytic solution, then adds the
    steady profile back. Single parameter set only -- this exists to check
    `pressure_head_iverson`, not to replace it.

    The internal grid starts at Z = 0 (where the boundary condition lives)
    and extends to `far_field_factor` times the deepest requested depth, so
    the zero-perturbation far-field condition does not contaminate the
    answer. Results are interpolated back onto `depths_m`.

    Returns
    -------
    psi : (n_t, n_z)
    """
    depths = np.atleast_1d(np.asarray(depths_m, dtype=float))
    edges = np.asarray(time_edges_s, dtype=float)
    intensity = np.asarray(intensity_ms, dtype=float)

    z_max = float(depths.max()) * far_field_factor
    grid = np.linspace(0.0, z_max, n_grid)
    dz = float(grid[1] - grid[0])

    cos2b = np.cos(slope_rad) ** 2
    d1 = diffusivity_m2s / cos2b

    u = np.zeros(n_grid, dtype=float)          # perturbation psi'
    out = np.zeros((edges.size - 1, depths.size), dtype=float)

    psi_s = steady_profile(depths, water_table_depth_m, slope_rad, steady_ratio)[0]

    for k in range(edges.size - 1):
        dt_total = float(edges[k + 1] - edges[k])
        flux_ratio = min(float(intensity[k]) / ksat_ms, 1.0)
        g = -flux_ratio                        # d(psi')/dZ at the surface
        dt = dt_total / n_sub
        r = d1 * dt / (2.0 * dz * dz)

        # Banded Crank-Nicolson matrix (constant within this interval)
        lower = np.full(n_grid, -r)
        main = np.full(n_grid, 1.0 + 2.0 * r)
        upper = np.full(n_grid, -r)
        # Surface: ghost node from the Neumann condition
        upper[0] = -2.0 * r
        # Far field: Dirichlet psi' = 0
        main[-1] = 1.0
        lower[-1] = 0.0

        ab = np.zeros((3, n_grid))
        ab[0, 1:] = upper[:-1]
        ab[1, :] = main
        ab[2, :-1] = lower[1:]

        for _ in range(n_sub):
            rhs = np.empty(n_grid)
            rhs[1:-1] = (
                (1.0 - 2.0 * r) * u[1:-1] + r * (u[2:] + u[:-2])
            )
            rhs[0] = (1.0 - 2.0 * r) * u[0] + 2.0 * r * u[1] - 4.0 * r * dz * g
            rhs[-1] = 0.0
            u = solve_banded((1, 1), ab, rhs)

        psi_full = np.interp(depths, grid, u) + psi_s
        out[k] = np.minimum(psi_full, depths)

    return out


def hyetograph_from_mm_per_hour(
    mm_per_hour: np.ndarray,
    interval_s: float = 3600.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a rainfall series in mm/h into solver inputs.

    Returns
    -------
    time_edges_s : (n + 1,)
    intensity_ms : (n,)
    """
    mmh = np.atleast_1d(np.asarray(mm_per_hour, dtype=float))
    intensity = mmh / 1000.0 / 3600.0
    edges = np.arange(mmh.size + 1, dtype=float) * interval_s
    return edges, intensity
