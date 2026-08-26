# 🏔️ Landslide Early Warning System (LEWS) — Northeast Region

An AI/ML-driven spatial early warning and incident management platform designed to monitor slope susceptibility, integrate live precipitation data, and empower citizens with real-time hazard crowdsourcing.

## 🚀 Live Demo
- **Web App**: [https://landslide-ews.onrender.com](https://landslide-ews.onrender.com)
- **API Documentation**: [https://landslide-ews.onrender.com/docs](https://landslide-ews.onrender.com/docs)

## 🛠️ Architecture & Tech Stack
- **Backend & API**: FastAPI, Python 3, SQLite / SQLAlchemy, Jinja2
- **Machine Learning**: Scikit-learn (Random Forest Classifier on slope, elevation & precipitation vectors)
- **Real-Time Weather**: Open-Meteo REST API
- **Frontend & GIS**: Leaflet.js, OpenStreetMap, Tailwind CSS
- **Deployment**: Render Cloud PaaS

## ⚡ Core Features
1. **Interactive Spatial Map**: Real-time color-coded risk assessment across key NER landslide corridors.
2. **Dynamic In-situ ML Scoring**: Incorporates real-time rainfall triggers with geomorphological factors.
3. **Crowdsourced Incident Reporting**: Geo-located citizen hazard pins visualized directly on the surveillance layer.
4. **Auto-fallback Pipeline**: Fault-tolerant model initialization and automatic static assets lifecycle.