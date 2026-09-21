import os
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.database import engine, Base
from app.migrate import reconcile_schema
from app.routes import weather, reports, risk, alerts, v2

Base.metadata.create_all(bind=engine)
# create_all() does not alter tables that already exist. The shipped
# landslide.db predates the current models, so reconcile before serving.
reconcile_schema(engine)

app = FastAPI(title="Landslide EWS API")

os.makedirs("app/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

templates = Jinja2Templates(directory="app/templates")

app.include_router(weather.router)
app.include_router(reports.router)
app.include_router(risk.router)
app.include_router(alerts.router)
app.include_router(v2.router)

@app.get("/")
def dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="dashboard.html")

@app.get("/report")
def report_page(request: Request):
    return templates.TemplateResponse(request=request, name="report.html")

@app.get("/api/health")
def health():
    return {"status": "ok", "message": "Landslide EWS API is running"}