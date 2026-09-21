# 🏔️ BHU-KAAL — Landslide Early Warning System for the North East

**SIH26001 · AI-Based Early Warning and Landslide Risk Monitoring System in NER · MDoNER**

*bhu* (earth) + *kaal* (time) — because the question that matters is not whether a slope is dangerous, but **how long you have**.

---

## The one-line pitch

> Every other landslide system predicts *whether* a slope is risky.
> We predict **how many hours of warning remain** — and we are the only ones modelling the fact that somebody cut that slope four months ago.

---

## Why this is different

| Layer | What it fixes | Status |
| --- | --- | --- |
| **A — Physics-generated labels** | India's landslide inventories are too small to train on. The best published Kerala study had **64 events since 2000** and an intensity–duration fit of **R² = 0.068**. | Monte Carlo infiltration + slope stability generates the training signal; ML is only a surrogate. |
| **B — Cut-slope disturbance** | NASA's LHASA v2 states in its own documentation that it *"does not incorporate any measures of anthropogenic disturbance"* and attributes its poor performance on disturbed terrain to exactly that. SIH26001 names "unplanned hill cutting" as a driver. | Disturbance modelled as a **decaying window of vulnerability**, feeding root cohesion in the physics — not bolted on as an opaque feature. |
| **C — Warning time** | Alerts say "high risk" but never say how long you have. | Rate-and-state friction gives χ, which classifies every slope into **synchronous / delayed / creep** and yields an estimated lead time. |
| **D — Cost-calibrated alerting** | ROC-optimal thresholds assume a missed landslide costs the same as a needless evacuation. Published work puts the ratio near **7:1**. | Thresholds minimise normalised expected cost, differ **per asset**, carry an interval, and emit **CAP 1.2** for SACHET / cell broadcast. |

### The finding that shapes the whole design

For slopes with **χ ≳ 4**, failure coincides with peak rainfall and there is no precursory movement to observe. An observation-driven warning system is *physically incapable* of helping. The only thing that works is acting on a rainfall **forecast**, ahead of the storm.

Cyclone Remal delivered 205 mm in 24 hours over Aizawl on 28 May 2024 and produced eight near-simultaneous failures. Run the hindcast — the system places it in exactly that regime:

```bash
curl localhost:8000/api/v2/hindcast/cyclone-remal
```

```
P(failure) = 0.70   interval [0.65, 0.75]   FS = 0.91 at 2.0 m
chi = 7.97          regime = synchronous    warning time = none
ALERT = EVACUATE    threshold = 0.0385 (hospital exposure)
```

---

## Quick start

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open <http://localhost:8000> for the dashboard, <http://localhost:8000/docs> for the API.

---

## API

### v2 — the BHU-KAAL engine

| Endpoint | What it gives you |
| --- | --- |
| `GET /api/v2/corridors` | Live assessment of all 18 monitored sites |
| `GET /api/v2/corridors/{id}` | Full four-layer breakdown for one site |
| `POST /api/v2/assess` | Assess any slope, with supplied or fetched rainfall |
| `GET /api/v2/corridors/{id}/cap.xml` | CAP 1.2 alert, ready for SACHET |
| `GET /api/v2/hindcast/cyclone-remal` | The held-out validation benchmark |
| `POST /api/v2/inverse-velocity` | Fukuzono countdown from a displacement series |
| `GET /api/v2/vulnerability-curve` | The disturbance recency curve |
| `GET /api/v2/regimes` | The three failure regimes explained |
| `GET /api/v2/thresholds` | Cost-derived threshold per asset class |
| `GET /api/v2/lithologies` | Geotechnical parameter library |

Add `?forecast_hours=24` to any corridor call to run in forecast mode — the only mode that can help a synchronous-regime slope.

### v1 — unchanged

`/api/health`, `/api/risk-zones/`, `/api/weather/{lat}/{lng}`, `/api/reports/`, `/api/alerts/` all behave as before.

---

## How a request flows

```
rainfall (IMD / IMERG / Open-Meteo)  ─┐
soil wetness (SMAP proxy)            ─┤
DEM slope, GSI lithology             ─┼──▶  Layer B: disturbance index
Sentinel-1/2 change detection        ─┘         │ (root cohesion, land cover)
                                                ▼
                                          Layer A: Monte Carlo
                                          infiltration + stability
                                                │ P(failure), ΔP, σ'
                                                ▼
                                          Layer C: χ → regime
                                                │ → hours of warning left
                                                ▼
                                          Layer D: cost threshold
                                          → alert + explanation + CAP 1.2
```

Runtime is roughly **25 ms per site**, so the API runs the physics live. No trained model is required at inference.

---

## Engine layout

```
app/engine/
├── physics/
│   ├── params.py          per-lithology geotechnical distributions (NER Tertiary units)
│   ├── infiltration.py    Iverson/TRIGRS analytic solver + Crank-Nicolson verifier
│   ├── stability.py       infinite-slope FS with root cohesion and surcharge
│   └── ensemble.py        Monte Carlo corpus generator
├── friction/
│   └── rate_state.py      χ, regimes, lead time, Fukuzono inverse velocity
├── disturbance/
│   └── recency.py         window of vulnerability, disturbance index
├── decision/
│   ├── cost.py            NEC thresholds, conformal intervals, asset classes
│   └── alerts.py          four-step ladder, explanations, CAP 1.2
├── evaluation/
│   └── metrics.py         POD/FAR/TSS/PR-AUC/Brier/lead-time, LOGO + temporal splits
├── sites.py               the monitored corridor registry
├── rainfall.py            hyetograph ingestion with labelled provenance
└── assess.py              the orchestrator
```

### Verification

The analytic infiltration solver is checked against an independent Crank–Nicolson finite-difference solution of the same problem. They agree to **0.000 m** of pressure head. Run:

```bash
python -c "
import numpy as np
from app.engine.physics.infiltration import *
d=np.linspace(0.4,4,10); e,i=hyetograph_from_mm_per_hour(np.full(24,100/24))
a=pressure_head_iverson(d,e,i,ksat_ms=1e-5,diffusivity_m2s=1e-4,slope_rad=np.deg2rad(35),water_table_depth_m=3.0)
n=pressure_head_numerical(d,e,i,ksat_ms=1e-5,diffusivity_m2s=1e-4,slope_rad=np.deg2rad(35),water_table_depth_m=3.0)
print('max abs diff:', abs(a[-1]-n[-1]).max())"
```

---

## Honest limitations

State these before a judge does.

- **Slope angles and lithology assignments in `sites.py` are representative values**, not surveyed ones. The correct upgrade is a lookup against a DEM and the GSI 1:50,000 geological map; the engine interface does not change when you do it.
- **Geotechnical parameters come from literature distributions**, not site investigation. That is precisely why every output is probabilistic with an explicit interval.
- **Disturbance records in `sites.py` are demonstration values.** The Sentinel-1 coherence / Sentinel-2 NDVI change-detection pipeline is specified but not yet wired in.
- **Satellite rainfall underestimates short intense convective cells**, and SMAP L4 carries roughly 2.5 days of latency. Gauge data matters in the NER.
- **`t_a` in the rate-and-state layer is a default**, not a fitted value. `calibrate_characteristic_time()` fits it against real events — do that as soon as you have NER events with known timing.
- **Alert thresholds are unvalidated.** At hospital exposure the threshold is 3.85%, which will produce frequent EVACUATE decisions. Thresholds are a policy input; tune the cost ratios with a district disaster manager and report the tuning.
- **CAP output ships as `status: Exercise`** and must stay that way until a competent authority authorises the system.

## Evaluation protocol

Do not report accuracy on a balanced split. Landslides are rare events on a daily grid; predicting "no landslide" forever scores above 99%.

- Hold out **spatially** (leave-one-basin-out) **and temporally** (train ≤ 2021, test 2022–2025). Doing only one lets leakage through.
- Report **POD, FAR, TSS, PR-AUC, Brier score**, and the **lead-time distribution** — the last of these is the actual early-warning metric and is essentially absent from the field.
- Keep Cyclone Remal out of training entirely and use it as the hindcast benchmark.
- Ship an **ablation**: baseline → + physics labels → + disturbance → + χ gating, reporting ΔTSS at each step.

`app/engine/evaluation/metrics.py` implements all of this, including `ablation_table()`.

---

## Roadmap

1. Replace `sites.py` terrain values with DEM + GSI raster lookups
2. Wire the Sentinel-1/Sentinel-2 change-detection pipeline into the disturbance layer
3. Ingest IMD gauge and IMERG rainfall in place of Open-Meteo
4. Ingest SMAP L4 soil wetness in place of the rainfall-derived proxy
5. Fit `t_a` and the alert thresholds against the GSI Bhukosh inventory
6. Train the XGBoost surrogate for grid-wide (rather than per-site) inference
7. NISAR L-band interferograms — free and public since 20 July 2026, 12-day repeat, and it penetrates the NER canopy where Sentinel-1 C-band decorrelates

---

## References

- LHASA v2, NASA/GSFC — global nowcasting architecture and its stated anthropogenic-disturbance gap
- *Frictional timescales and the impact of climate change-driven extreme weather on rainfall-triggered landslides in Mizoram, NE India* (2026), arXiv:2606.23281 — the χ parameter and the three regimes
- *A hybrid physics–machine learning framework for probabilistic landslide prediction based on critical rainfall envelopes* (2026) — the surrogate design
- *Cost-sensitive rainfall thresholds for shallow landslides*, Landslides (2021) — normalised expected cost
- Iverson (2000), WRR 36(7); Baum, Savage & Godt (2008), USGS OFR 2008-1159 — the infiltration solution
- Fukuzono (1985) — inverse-velocity failure prediction
- *Deep learning for potential landslide identification*, NHESS review (2026) — the interpretability and generalisation gaps
- NDMA SACHET / CAP 1.2 — alert dissemination

---

## Tech stack

FastAPI · SQLAlchemy · NumPy · SciPy · Jinja2 · Leaflet.js · scikit-learn (v1 model) · Render
