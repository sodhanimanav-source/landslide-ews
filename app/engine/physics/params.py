"""
Geotechnical parameter distributions by lithology.

These distributions are what let us run a Monte Carlo ensemble instead of
pretending we know the shear strength of every hillside in the North East.
Each lithology carries plausible ranges from the engineering-geology
literature for residual and colluvial soils developed on that parent rock.

Lithology codes follow the Tertiary sequence that actually underlies the
NER hill states (Barail, Surma, Tipam, Disang) plus basement and cover units.

IMPORTANT: these are literature priors, not site investigations. Every
downstream product is probabilistic for exactly this reason. If your team
obtains real borehole or lab data for a study site, override the entry here
and say so in your report -- that is a strength, not an admission.

References for typical ranges:
  - Iverson (2000), Landslide triggering by rain infiltration, WRR 36(7)
  - Baum, Savage & Godt (2008), TRIGRS v2, USGS OFR 2008-1159
  - Standard residual-soil parameter ranges for tropical weathered profiles
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np

# Unit weight of water, kN/m^3
GAMMA_W = 9.81


@dataclass(frozen=True)
class LithologyParams:
    """Parameter distributions for one lithology class.

    Strength parameters are drawn from truncated normals; conductivity and
    diffusivity are drawn lognormally, because they vary over orders of
    magnitude and cannot go negative.

    Attributes
    ----------
    name : human-readable lithology name
    cohesion_kpa : (mean, sd) effective cohesion c', kPa
    friction_deg : (mean, sd) effective friction angle phi', degrees
    unit_weight_knm3 : (mean, sd) soil unit weight gamma_s, kN/m^3
    ksat_log10_ms : (mean, sd) log10 saturated conductivity, m/s
    diffusivity_log10_m2s : (mean, sd) log10 hydraulic diffusivity D0, m^2/s
    soil_depth_m : (mean, sd) depth to basal boundary, m
    strength_rank : 0-7 ordinal rock strength, after LHASA v2's lithology term
    """

    name: str
    cohesion_kpa: tuple[float, float]
    friction_deg: tuple[float, float]
    unit_weight_knm3: tuple[float, float]
    ksat_log10_ms: tuple[float, float]
    diffusivity_log10_m2s: tuple[float, float]
    soil_depth_m: tuple[float, float]
    strength_rank: int

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# NER lithology library
# ---------------------------------------------------------------------------
# Barail / Surma / Tipam / Disang are the real Tertiary units underlying
# Mizoram, southern Assam and parts of Manipur and Nagaland. Aizawl -- our
# anchor validation site -- sits almost entirely on Surma Group sandstone
# and shale, so that entry deserves the most scrutiny if you get real data.

LITHOLOGY_LIBRARY: Dict[str, LithologyParams] = {
    "surma_shale_sandstone": LithologyParams(
        name="Surma Group (shale-sandstone, Aizawl)",
        cohesion_kpa=(6.0, 3.0),
        friction_deg=(29.0, 4.0),
        unit_weight_knm3=(19.0, 1.0),
        ksat_log10_ms=(-5.3, 0.6),
        diffusivity_log10_m2s=(-4.6, 0.6),
        soil_depth_m=(2.2, 0.8),
        strength_rank=3,
    ),
    "barail_sandstone": LithologyParams(
        name="Barail Group (sandstone-dominant)",
        cohesion_kpa=(8.0, 3.5),
        friction_deg=(32.0, 4.0),
        unit_weight_knm3=(19.5, 1.0),
        ksat_log10_ms=(-4.9, 0.6),
        diffusivity_log10_m2s=(-4.2, 0.6),
        soil_depth_m=(2.0, 0.8),
        strength_rank=4,
    ),
    "tipam_sandstone": LithologyParams(
        name="Tipam Group (friable sandstone)",
        cohesion_kpa=(4.0, 2.5),
        friction_deg=(31.0, 4.0),
        unit_weight_knm3=(18.5, 1.0),
        ksat_log10_ms=(-4.5, 0.6),
        diffusivity_log10_m2s=(-3.9, 0.6),
        soil_depth_m=(2.5, 0.9),
        strength_rank=3,
    ),
    "disang_shale": LithologyParams(
        name="Disang Group (shale, highly sheared)",
        cohesion_kpa=(5.0, 3.0),
        friction_deg=(26.0, 4.0),
        unit_weight_knm3=(19.0, 1.0),
        ksat_log10_ms=(-6.0, 0.7),
        diffusivity_log10_m2s=(-5.2, 0.7),
        soil_depth_m=(1.8, 0.7),
        strength_rank=2,
    ),
    "gneiss_granite": LithologyParams(
        name="Basement gneiss / granite",
        cohesion_kpa=(12.0, 5.0),
        friction_deg=(35.0, 4.0),
        unit_weight_knm3=(20.0, 1.0),
        ksat_log10_ms=(-5.5, 0.8),
        diffusivity_log10_m2s=(-4.8, 0.8),
        soil_depth_m=(1.5, 0.7),
        strength_rank=6,
    ),
    "quartzite": LithologyParams(
        name="Quartzite / metasediment",
        cohesion_kpa=(14.0, 5.0),
        friction_deg=(36.0, 4.0),
        unit_weight_knm3=(20.5, 1.0),
        ksat_log10_ms=(-5.8, 0.8),
        diffusivity_log10_m2s=(-5.0, 0.8),
        soil_depth_m=(1.2, 0.6),
        strength_rank=7,
    ),
    "laterite": LithologyParams(
        name="Laterite / ferricrete cover",
        cohesion_kpa=(10.0, 4.0),
        friction_deg=(30.0, 4.0),
        unit_weight_knm3=(18.0, 1.0),
        ksat_log10_ms=(-4.6, 0.7),
        diffusivity_log10_m2s=(-4.0, 0.7),
        soil_depth_m=(2.0, 0.8),
        strength_rank=4,
    ),
    "colluvium": LithologyParams(
        name="Colluvium / residual soil (default)",
        cohesion_kpa=(3.0, 2.0),
        friction_deg=(28.0, 4.0),
        unit_weight_knm3=(18.0, 1.0),
        ksat_log10_ms=(-4.8, 0.7),
        diffusivity_log10_m2s=(-4.1, 0.7),
        soil_depth_m=(2.8, 1.0),
        strength_rank=1,
    ),
    "alluvium": LithologyParams(
        name="Alluvium / valley fill",
        cohesion_kpa=(5.0, 3.0),
        friction_deg=(30.0, 4.0),
        unit_weight_knm3=(18.5, 1.0),
        ksat_log10_ms=(-4.2, 0.7),
        diffusivity_log10_m2s=(-3.6, 0.7),
        soil_depth_m=(4.0, 1.5),
        strength_rank=1,
    ),
}

DEFAULT_LITHOLOGY = "colluvium"


def get_lithology(code: str) -> LithologyParams:
    """Look up a lithology, falling back to colluvium for anything unmapped.

    Falling back to the weakest common unit is deliberate: an unknown
    lithology should not silently become a strong one.
    """
    return LITHOLOGY_LIBRARY.get(code, LITHOLOGY_LIBRARY[DEFAULT_LITHOLOGY])


def sample_parameters(
    lithology: str,
    n: int,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Draw `n` parameter sets for a lithology.

    Strength parameters are truncated at physically sensible floors so a
    tail draw cannot produce negative cohesion or a frictionless soil.

    Returns a dict of arrays, each of length `n`:
        cohesion_kpa, friction_rad, unit_weight_knm3, ksat_ms,
        diffusivity_m2s, soil_depth_m
    """
    if rng is None:
        rng = np.random.default_rng()

    lp = get_lithology(lithology)

    cohesion = np.clip(rng.normal(*lp.cohesion_kpa, size=n), 0.0, None)
    friction = np.clip(rng.normal(*lp.friction_deg, size=n), 12.0, 50.0)
    unit_weight = np.clip(rng.normal(*lp.unit_weight_knm3, size=n), 14.0, 24.0)
    ksat = 10.0 ** rng.normal(*lp.ksat_log10_ms, size=n)
    diffusivity = 10.0 ** rng.normal(*lp.diffusivity_log10_m2s, size=n)
    depth = np.clip(rng.normal(*lp.soil_depth_m, size=n), 0.4, None)

    # Physical consistency: hydraulic diffusivity should exceed conductivity
    # for these units (D0 = K / specific storage, and storage << 1).
    diffusivity = np.maximum(diffusivity, ksat * 2.0)

    return {
        "cohesion_kpa": cohesion,
        "friction_rad": np.deg2rad(friction),
        "unit_weight_knm3": unit_weight,
        "ksat_ms": ksat,
        "diffusivity_m2s": diffusivity,
        "soil_depth_m": depth,
    }


def lithology_codes() -> list[str]:
    """All lithology codes, for iterating the ensemble over map units."""
    return list(LITHOLOGY_LIBRARY.keys())
