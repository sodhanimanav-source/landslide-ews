"""
Site registry: the NER corridors and urban zones the system assesses.

Each entry carries the terrain attributes the physics engine needs -- a
representative slope angle, a lithology from the NER Tertiary sequence, and
the exposed asset class that sets its cost-based alert threshold.

Provenance note, which belongs in your report
---------------------------------------------
Slope angles and lithology assignments here are REPRESENTATIVE values for
each corridor, assigned from published regional geology. They are
placeholders for a proper terrain-unit layer derived from a DEM and the GSI
1:50,000 geological map. The right upgrade is to replace this module with a
lookup against those rasters; the engine interface does not change when you
do. Say this plainly rather than implying every value is surveyed.

Before this module existed, the dashboard carried hardcoded risk strings
like "CRITICAL RED (91%)" that were never computed from anything. Those
numbers are now produced by the engine from live rainfall.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Sequence

from .disturbance.recency import DisturbanceRecord, DisturbanceType


@dataclass(frozen=True)
class Site:
    """A monitored location."""

    id: str
    name: str
    state: str
    latitude: float
    longitude: float
    slope_deg: float
    lithology: str
    asset_class: str = "state_road"
    category: str = ""
    surcharge_kpa: float = 0.0
    # Known disturbances. In production these come from the Sentinel-1 /
    # Sentinel-2 change-detection pipeline, not from a literal here.
    disturbances: tuple[DisturbanceRecord, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "state": self.state,
            "lat": self.latitude,
            "lng": self.longitude,
            "slope_deg": self.slope_deg,
            "lithology": self.lithology,
            "asset_class": self.asset_class,
            "category": self.category,
        }


# Demonstration disturbance records. Replace with pipeline output.
_AIZAWL_CUT = DisturbanceRecord(
    detected_on=date(2026, 4, 15),
    kind=DisturbanceType.ROAD_CUT,
    confidence=0.88,
    area_fraction=0.55,
    source="sentinel1_coherence (demo)",
)
_GANGTOK_CUT = DisturbanceRecord(
    detected_on=date(2025, 11, 2),
    kind=DisturbanceType.BUILDING_CUT,
    confidence=0.80,
    area_fraction=0.45,
    source="sentinel2_ndvi (demo)",
)
_NONEY_CLEAR = DisturbanceRecord(
    detected_on=date(2025, 7, 20),
    kind=DisturbanceType.FOREST_LOSS,
    confidence=0.75,
    area_fraction=0.60,
    source="hansen_gfc (demo)",
)


SITES: tuple[Site, ...] = (
    Site(
        id="nh10-sevoke-gangtok",
        name="Sikkim: NH-10 (Sevoke - Gangtok)",
        state="Sikkim",
        latitude=27.050, longitude=88.435,
        slope_deg=42.0, lithology="gneiss_granite",
        asset_class="national_highway",
        category="Arterial corridor",
    ),
    Site(
        id="chungthang-mangan",
        name="North Sikkim: Chungthang - Mangan Belt",
        state="Sikkim",
        latitude=27.600, longitude=88.640,
        slope_deg=45.0, lithology="gneiss_granite",
        asset_class="state_road",
        category="GLOF and moraine zone",
    ),
    Site(
        id="sela-tawang",
        name="Arunachal: Sela Pass - Tawang Corridor",
        state="Arunachal Pradesh",
        latitude=27.505, longitude=92.100,
        slope_deg=43.0, lithology="quartzite",
        asset_class="national_highway",
        category="High-altitude strategic pass",
    ),
    Site(
        id="itanagar-ziro",
        name="Arunachal: Trans-Arunachal (Itanagar - Ziro)",
        state="Arunachal Pradesh",
        latitude=27.350, longitude=93.750,
        slope_deg=32.0, lithology="tipam_sandstone",
        asset_class="national_highway",
        category="Highway valley stretch",
    ),
    Site(
        id="nh29-kohima",
        name="Nagaland: NH-29 (Chumukedima - Kohima)",
        state="Nagaland",
        latitude=25.710, longitude=93.980,
        slope_deg=38.0, lithology="disang_shale",
        asset_class="national_highway",
        category="Fragile shale slopes",
    ),
    Site(
        id="lumding-badarpur",
        name="Assam Dima Hasao: Lumding - Badarpur Hill Rail",
        state="Assam",
        latitude=25.150, longitude=93.100,
        slope_deg=36.0, lithology="barail_sandstone",
        asset_class="national_highway",
        category="Critical rail infrastructure",
    ),
    Site(
        id="nh6-sonapur",
        name="Meghalaya: NH-6 Sonapur Tunnel Zone",
        state="Meghalaya",
        latitude=25.200, longitude=92.350,
        slope_deg=40.0, lithology="quartzite",
        asset_class="national_highway",
        category="Heavy-rainfall tunnel corridor",
    ),
    Site(
        id="cherrapunji-mawsynram",
        name="Meghalaya: Cherrapunji - Mawsynram Escarpment",
        state="Meghalaya",
        latitude=25.2986, longitude=91.7302,
        slope_deg=35.0, lithology="quartzite",
        asset_class="settlement",
        category="Extreme precipitation plateau",
    ),
    Site(
        id="nh37-imphal-jiribam",
        name="Manipur: NH-37 (Imphal - Jiribam Highway)",
        state="Manipur",
        latitude=24.780, longitude=93.500,
        slope_deg=37.0, lithology="disang_shale",
        asset_class="national_highway",
        category="Mudflow and debris corridor",
    ),
    Site(
        id="tupul-noney",
        name="Manipur: Tupul - Noney Railway Basin",
        state="Manipur",
        latitude=24.710, longitude=93.650,
        slope_deg=41.0, lithology="disang_shale",
        asset_class="national_highway",
        category="Geological fracture zone",
        disturbances=(_NONEY_CLEAR,),
    ),
    Site(
        id="nh54-aizawl-lunglei",
        name="Mizoram: NH-54 (Aizawl - Lunglei Axis)",
        state="Mizoram",
        latitude=23.727, longitude=92.717,
        slope_deg=39.0, lithology="surma_shale_sandstone",
        asset_class="national_highway",
        category="Urban ridge and valley highway",
        surcharge_kpa=15.0,
        disturbances=(_AIZAWL_CUT,),
    ),
    Site(
        id="jampui-hills",
        name="Tripura: Jampui Hills Ridge",
        state="Tripura",
        latitude=23.950, longitude=92.280,
        slope_deg=25.0, lithology="tipam_sandstone",
        asset_class="state_road",
        category="Sedimentary formation",
    ),
    Site(
        id="shillong-bypass",
        name="Meghalaya: Shillong Bypass NH-206",
        state="Meghalaya",
        latitude=25.350, longitude=91.950,
        slope_deg=22.0, lithology="gneiss_granite",
        asset_class="national_highway",
        category="Stable granite terrain",
    ),
    # Urban zones -- higher-value exposure, lower alert thresholds.
    Site(
        id="gangtok-urban",
        name="Gangtok Urban Slopes",
        state="Sikkim",
        latitude=27.3389, longitude=88.6065,
        slope_deg=40.0, lithology="gneiss_granite",
        asset_class="settlement",
        category="Dense construction on steep slope cuts",
        surcharge_kpa=25.0,
        disturbances=(_GANGTOK_CUT,),
    ),
    Site(
        id="haflong",
        name="Haflong Hill Station",
        state="Assam",
        latitude=25.1833, longitude=93.0167,
        slope_deg=33.0, lithology="barail_sandstone",
        asset_class="settlement",
        category="Subsidence and rotational slump",
        surcharge_kpa=18.0,
    ),
    Site(
        id="aizawl-ridge",
        name="Aizawl Urban Ridge",
        state="Mizoram",
        latitude=23.7271, longitude=92.7176,
        slope_deg=42.0, lithology="surma_shale_sandstone",
        asset_class="hospital",
        category="Fragile shale, surface water infiltration",
        surcharge_kpa=30.0,
        disturbances=(_AIZAWL_CUT,),
    ),
    Site(
        id="tawang-plateau",
        name="Tawang High Plateau",
        state="Arunachal Pradesh",
        latitude=27.5861, longitude=91.8653,
        slope_deg=38.0, lithology="quartzite",
        asset_class="settlement",
        category="Freeze-thaw weathering, moraine debris",
    ),
    Site(
        id="kohima-ridges",
        name="Kohima Town Ridges",
        state="Nagaland",
        latitude=25.6751, longitude=94.1086,
        slope_deg=36.0, lithology="disang_shale",
        asset_class="settlement",
        category="Road sinking and creeping deformation",
        surcharge_kpa=20.0,
    ),
)

SITES_BY_ID: dict[str, Site] = {s.id: s for s in SITES}


def get_site(site_id: str) -> Site | None:
    return SITES_BY_ID.get(site_id)


def all_sites() -> Sequence[Site]:
    return SITES
