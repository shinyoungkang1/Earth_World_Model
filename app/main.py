"""FastAPI app for the SubTerra PA MVP map viewer."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.datastore import datastore, parse_bbox
from app.planet import planet_client


app = FastAPI(title="SubTerra PA MVP")
static_dir = Path(settings()["repo_root"]) / "app" / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.on_event("startup")
def warm_cache() -> None:
    datastore()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/summary")
def summary() -> dict:
    return datastore().summary()


@app.get("/api/layers")
def layers() -> dict:
    return {"layers": datastore().layer_catalog()}


@app.get("/api/composites")
def composites() -> dict:
    return {"composites": datastore().composites()}


@app.get("/api/composites/{composite_id}/image")
def composite_image(composite_id: str) -> FileResponse:
    image_path = datastore().composite_path(composite_id)
    if image_path is None or not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Unknown composite {composite_id}")
    return FileResponse(image_path)


@app.get("/api/geojson/{layer_id}")
def geojson_layer(
    layer_id: str,
    bbox: str | None = Query(default=None),
    limit: int = Query(default=1000, ge=1, le=20000),
):
    try:
        bbox_tuple = parse_bbox(bbox)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    store = datastore()
    if layer_id == "wells":
        return JSONResponse(store.wells_geojson(bbox_tuple, limit))
    if layer_id == "permits":
        return JSONResponse(store.permits_geojson(bbox_tuple, limit))
    if layer_id == "geology":
        return JSONResponse(store.geology_geojson(bbox_tuple, min(limit, 2500)))
    if layer_id == "sentinel_scenes":
        return JSONResponse(store.sentinel_geojson(bbox_tuple, limit))
    if layer_id == "landsat_scenes":
        return JSONResponse(store.landsat_geojson(bbox_tuple, limit))
    if layer_id == "planet_scenes":
        return JSONResponse(store.planet_geojson(bbox_tuple, min(limit, 1000)))
    if layer_id == "gas_prospect_cells_v0":
        return JSONResponse(store.gas_prospect_geojson(bbox_tuple, min(limit, 10000)))
    if layer_id == "gas_prospect_cells_v1":
        return JSONResponse(store.gas_prospect_v1_geojson(bbox_tuple, min(limit, 10000)))
    if layer_id == "gas_prospect_cells_prithvi_v1":
        return JSONResponse(store.gas_prospect_prithvi_v1_geojson(bbox_tuple, min(limit, 10000)))
    raise HTTPException(status_code=404, detail=f"Unknown layer {layer_id}")


@app.get("/api/prospects/gas_v1/top")
def gas_v1_top_prospects(limit: int = Query(default=20, ge=1, le=100)):
    return {"prospects": datastore().top_prospects_v1(limit)}


@app.get("/api/models/gas_v1")
def gas_v1_model_summary() -> dict:
    return datastore().gas_model_v1_summary()


@app.get("/api/prospects/prithvi_v1/top")
def gas_prithvi_v1_top_prospects(limit: int = Query(default=20, ge=1, le=100)):
    return {"prospects": datastore().top_prospects_prithvi_v1(limit)}


@app.get("/api/models/prithvi_v1")
def gas_prithvi_v1_model_summary() -> dict:
    return datastore().gas_model_prithvi_v1_summary()


@app.get("/api/planet/status")
def planet_status() -> dict:
    return planet_client().status()


@app.get("/api/planet/mosaics")
def planet_mosaics() -> dict:
    status = planet_client().status()
    if not status.get("authorized"):
        return {"mosaics": [], "status": status}
    return {"status": status, **planet_client().list_mosaics()}


@app.get("/api/planet/tiles/{mosaic_name}/{z}/{x}/{y}.png")
def planet_tiles(mosaic_name: str, z: int, x: int, y: int):
    status = planet_client().status()
    if not status.get("authorized"):
        raise HTTPException(status_code=502, detail=status["message"])
    try:
        body, content_type = planet_client().fetch_tile(mosaic_name, z, x, y)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return Response(content=body, media_type=content_type)
