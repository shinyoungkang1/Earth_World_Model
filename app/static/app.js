const map = new maplibregl.Map({
  container: "map",
  style: {
    version: 8,
    sources: {
      osm: {
        type: "raster",
        tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
        tileSize: 256,
        attribution: "© OpenStreetMap contributors",
      },
    },
    layers: [{ id: "osm", type: "raster", source: "osm" }],
  },
  center: [-77.8, 40.9],
  zoom: 6.3,
});

const layerStyles = {
  wells: {
    type: "circle",
    paint: { "circle-radius": 2.5, "circle-color": "#1d5f5a", "circle-opacity": 0.8 },
  },
  permits: {
    type: "circle",
    paint: { "circle-radius": 3.5, "circle-color": "#b8662f", "circle-stroke-width": 1, "circle-stroke-color": "#fff5e9" },
  },
  geology: {
    type: "fill",
    paint: { "fill-color": "#d8c58b", "fill-opacity": 0.22, "fill-outline-color": "#6d6549" },
  },
  sentinel_scenes: {
    type: "line",
    paint: { "line-color": "#355ff0", "line-width": 1.25, "line-opacity": 0.75 },
  },
  landsat_scenes: {
    type: "line",
    paint: { "line-color": "#d946ef", "line-width": 1.25, "line-opacity": 0.75 },
  },
  planet_scenes: {
    type: "line",
    paint: { "line-color": "#111827", "line-width": 1.35, "line-opacity": 0.72 },
  },
  gas_prospect_cells_v0: {
    type: "fill",
    paint: {
      "fill-color": [
        "interpolate",
        ["linear"],
        ["coalesce", ["get", "score"], 0],
        0.15, "#f0e8c9",
        0.35, "#d4cd7a",
        0.5, "#8fbc5a",
        0.65, "#3f965d",
        0.8, "#1f6d5b",
        0.95, "#153a45",
      ],
      "fill-opacity": 0.48,
      "fill-outline-color": "#24454e",
    },
  },
  gas_prospect_cells_v1: {
    type: "fill",
    paint: {
      "fill-color": [
        "interpolate",
        ["linear"],
        ["coalesce", ["get", "score"], 0],
        0.05, "#f6f1da",
        0.25, "#d7c96d",
        0.5, "#8ab257",
        0.75, "#2d8c63",
        0.9, "#145c52",
        0.99, "#0b2d34",
      ],
      "fill-opacity": 0.58,
      "fill-outline-color": "#0f3b42",
    },
  },
  gas_prospect_cells_prithvi_v1: {
    type: "fill",
    paint: {
      "fill-color": [
        "interpolate",
        ["linear"],
        ["coalesce", ["get", "score"], 0],
        0.05, "#f5f4ec",
        0.2, "#d8d0a9",
        0.4, "#a8b66b",
        0.6, "#5b9b67",
        0.8, "#1f6b61",
        0.95, "#0d2632",
      ],
      "fill-opacity": 0.62,
      "fill-outline-color": "#173842",
    },
  },
};

const popupFields = {
  wells: ["well_api", "operator_name", "well_status", "well_type", "county_name"],
  permits: ["authorization_id", "operator", "permit_issued_date", "spud_date", "county"],
  geology: ["name", "age", "lith1", "l_desc"],
  sentinel_scenes: ["scene_id", "datetime", "cloud_cover", "platform"],
  landsat_scenes: ["scene_id", "datetime", "cloud_cover", "platform"],
  planet_scenes: ["scene_id", "acquired", "cloud_cover", "clear_percent", "quality_category", "satellite_id"],
  gas_prospect_cells_v0: [
    "score",
    "score_percentile",
    "prospect_tier",
    "score_rank",
    "county_name",
    "municipality_name",
    "geology_name",
    "geology_lith1",
    "nearest_well_distance_km",
  ],
  gas_prospect_cells_v1: [
    "score",
    "score_percentile",
    "prospect_tier",
    "score_rank",
    "county_name",
    "geology_name",
    "geology_lith1",
    "elevation_m",
    "slope_deg",
    "fault_distance_km",
    "well_count_5km",
    "permit_count_5km",
  ],
  gas_prospect_cells_prithvi_v1: [
    "score",
    "score_percentile",
    "prospect_tier",
    "score_rank",
    "county_name",
    "geology_name",
    "geology_lith1",
    "fault_distance_km",
    "well_count_5km",
    "permit_count_5km",
  ],
};

const popupLabels = {
  well_api: "Well API",
  operator_name: "Operator",
  well_status: "Status",
  well_type: "Type",
  county_name: "County",
  municipality_name: "Municipality",
  authorization_id: "Authorization",
  permit_issued_date: "Permit issued",
  spud_date: "Spud date",
  county: "County",
  name: "Unit",
  age: "Age",
  lith1: "Lithology",
  l_desc: "Description",
  scene_id: "Scene",
  acquired: "Acquired",
  datetime: "Datetime",
  cloud_cover: "Cloud cover",
  platform: "Platform",
  clear_percent: "Clear percent",
  quality_category: "Quality",
  satellite_id: "Satellite",
  score: "Score",
  score_percentile: "Percentile",
  prospect_tier: "Tier",
  score_rank: "Rank",
  geology_name: "Geology",
  geology_lith1: "Lithology",
  nearest_well_distance_km: "Nearest well km",
  elevation_m: "Elevation m",
  slope_deg: "Slope deg",
  fault_distance_km: "Fault dist km",
  well_count_5km: "Wells in 5 km",
  permit_count_5km: "Permits in 5 km",
};

const popup = new maplibregl.Popup({ closeButton: false, closeOnClick: false });
const popupBoundLayers = new Set();
const compositeCatalog = new Map();

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`${res.status} ${res.statusText}`);
  }
  return res.json();
}

function currentBbox() {
  const bounds = map.getBounds();
  return [bounds.getWest(), bounds.getSouth(), bounds.getEast(), bounds.getNorth()].join(",");
}

async function loadSummary() {
  const summary = await fetchJson("/api/summary");
  const root = document.getElementById("summary");
  root.innerHTML = "";
  const items = [
    ["Wells", summary.wells],
    ["Permits", summary.permits],
    ["Production rows", summary.production_records],
    ["Bedrock polygons", summary.bedrock_polygons],
    ["Sentinel scenes", summary.sentinel_scenes],
    ["Landsat scenes", summary.landsat_scenes],
    ["Planet scenes", summary.planet_scenes],
    ["Prospect cells", summary.gas_prospect_cells_v0],
    ["Prospect cells v1", summary.gas_prospect_cells_v1],
    ["Prospect cells EO", summary.gas_prospect_cells_prithvi_v1],
    ["Raster assets", summary.raster_assets],
    ["Raster composites", summary.raster_composites],
  ];
  for (const [label, value] of items) {
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.textContent = value;
    root.append(dt, dd);
  }
}

async function loadGasModelInfo() {
  const payload = await fetchJson("/api/models/gas_v1");
  const basin = payload.basin || {};
  const metrics = payload.metrics || {};
  const layer = payload.prospect_layer || {};
  const root = document.getElementById("model-status");
  if (!root) {
    return;
  }
  if (!metrics.label_column) {
    root.textContent = "No v1 model available yet.";
    return;
  }
  root.innerHTML = `
    <div><strong>${basin.display_name || "Gas v1"}</strong></div>
    <div>Label: ${metrics.label_column}</div>
    <div>ROC-AUC: ${Number(metrics.roc_auc_test || 0).toFixed(3)}</div>
    <div>AP: ${Number(metrics.average_precision_test || 0).toFixed(3)}</div>
    <div>Holdout cutoff: ${metrics.temporal_holdout_cutoff_date || "n/a"}</div>
    <div>Grid cells: ${layer.row_count || 0}</div>
  `;
}

async function loadTopProspects() {
  const payload = await fetchJson("/api/prospects/gas_v1/top?limit=12");
  const root = document.getElementById("top-prospects");
  if (!root) {
    return;
  }
  root.innerHTML = "";
  if (!payload.prospects?.length) {
    root.textContent = "No v1 prospects available yet.";
    return;
  }
  for (const item of payload.prospects) {
    const row = document.createElement("div");
    row.className = "prospect-row";
    row.innerHTML = `
      <div class="prospect-rank">#${item.score_rank}</div>
      <div class="prospect-main">
        <div><strong>${item.county_name || "Unknown county"}</strong> | ${(item.prospect_tier || "").replace("_", " ")}</div>
        <div class="muted">Score ${Number(item.score).toFixed(3)} | ${item.geology_name || "No geology"}</div>
      </div>
    `;
    root.appendChild(row);
  }
}

async function loadPlanetStatus() {
  const status = await fetchJson("/api/planet/status");
  const box = document.getElementById("planet-status");
  const select = document.getElementById("planet-mosaics");
  if (!status.configured) {
    box.textContent = "Planet key not configured.";
    select.disabled = true;
    return;
  }
  if (!status.authorized) {
    box.textContent = status.message;
    select.disabled = true;
    return;
  }
  box.textContent = "Planet basemaps available.";
  const payload = await fetchJson("/api/planet/mosaics");
  select.innerHTML = "";
  for (const mosaic of payload.mosaics) {
    const option = document.createElement("option");
    option.value = mosaic.name;
    option.textContent = mosaic.name;
    select.append(option);
  }
  select.disabled = payload.mosaics.length === 0;
}

async function loadCompositeCatalog() {
  const payload = await fetchJson("/api/composites");
  for (const composite of payload.composites) {
    compositeCatalog.set(composite.composite_id, composite);
  }
  for (const input of document.querySelectorAll("input[data-composite]")) {
    input.disabled = !compositeCatalog.has(input.dataset.composite);
  }
}

function ensureSource(layerId) {
  if (!map.getSource(layerId)) {
    map.addSource(layerId, { type: "geojson", data: { type: "FeatureCollection", features: [] } });
  }
  if (!map.getLayer(layerId)) {
    map.addLayer({ id: layerId, source: layerId, ...layerStyles[layerId] });
    bindPopup(layerId);
  }
}

function featurePopupHtml(layerId, properties) {
  const fields = popupFields[layerId] || [];
  const lines = [];
  for (const field of fields) {
    const value = properties?.[field];
    if (value === null || value === undefined || value === "") {
      continue;
    }
    const label = popupLabels[field] || field;
    lines.push(`<div><strong>${label}:</strong> ${value}</div>`);
  }
  return lines.join("");
}

function bindPopup(layerId) {
  if (popupBoundLayers.has(layerId)) {
    return;
  }
  popupBoundLayers.add(layerId);
  map.on("mouseenter", layerId, () => {
    map.getCanvas().style.cursor = "pointer";
  });
  map.on("mouseleave", layerId, () => {
    map.getCanvas().style.cursor = "";
    popup.remove();
  });
  map.on("click", layerId, (event) => {
    const feature = event.features?.[0];
    if (!feature) {
      return;
    }
    popup
      .setLngLat(event.lngLat)
      .setHTML(featurePopupHtml(layerId, feature.properties))
      .addTo(map);
  });
}

async function refreshLayer(layerId) {
  ensureSource(layerId);
  const bbox = currentBbox();
  const limit = layerId === "geology" ? 1200 : 2500;
  const finalLimit =
    layerId === "gas_prospect_cells_v0" ||
    layerId === "gas_prospect_cells_v1" ||
    layerId === "gas_prospect_cells_prithvi_v1"
      ? 10000
      : limit;
  const payload = await fetchJson(`/api/geojson/${layerId}?bbox=${encodeURIComponent(bbox)}&limit=${finalLimit}`);
  map.getSource(layerId).setData(payload);
  const checked = document.querySelector(`input[data-layer="${layerId}"]`).checked;
  map.setLayoutProperty(layerId, "visibility", checked ? "visible" : "none");
  return payload.meta;
}

function setPlanetBase(enabled) {
  const existing = map.getLayer("planet-base");
  if (existing) {
    map.removeLayer("planet-base");
  }
  if (map.getSource("planet-base")) {
    map.removeSource("planet-base");
  }
  if (!enabled) {
    return;
  }
  const mosaic = document.getElementById("planet-mosaics").value;
  if (!mosaic) {
    return;
  }
  map.addSource("planet-base", {
    type: "raster",
    tiles: [`/api/planet/tiles/${mosaic}/{z}/{x}/{y}.png`],
    tileSize: 256,
  });
  map.addLayer({ id: "planet-base", type: "raster", source: "planet-base" }, "osm");
}

function compositeCoordinates(composite) {
  return [
    [composite.bbox_west, composite.bbox_north],
    [composite.bbox_east, composite.bbox_north],
    [composite.bbox_east, composite.bbox_south],
    [composite.bbox_west, composite.bbox_south],
  ];
}

function ensureCompositeLayer(compositeId) {
  const composite = compositeCatalog.get(compositeId);
  if (!composite) {
    return false;
  }
  if (!map.getSource(compositeId)) {
    map.addSource(compositeId, {
      type: "image",
      url: composite.image_url,
      coordinates: compositeCoordinates(composite),
    });
  }
  if (!map.getLayer(compositeId)) {
    const layerDef = {
      id: compositeId,
      type: "raster",
      source: compositeId,
      paint: { "raster-opacity": 0.78 },
    };
    if (map.getLayer("wells")) {
      map.addLayer(layerDef, "wells");
    } else {
      map.addLayer(layerDef);
    }
  }
  return true;
}

function refreshVisibleComposites() {
  const status = [];
  for (const input of document.querySelectorAll("input[data-composite]")) {
    const compositeId = input.dataset.composite;
    if (!compositeCatalog.has(compositeId)) {
      input.disabled = true;
      continue;
    }
    ensureCompositeLayer(compositeId);
    map.setLayoutProperty(compositeId, "visibility", input.checked ? "visible" : "none");
    if (input.checked) {
      status.push(compositeId);
    }
  }
  return status;
}

async function refreshVisibleLayers() {
  const status = [];
  const tasks = [];
  document.getElementById("layer-status").textContent = "Loading visible layers...";
  for (const input of document.querySelectorAll("input[data-layer]")) {
    if (!input.checked) {
      if (map.getLayer(input.dataset.layer)) {
        map.setLayoutProperty(input.dataset.layer, "visibility", "none");
      }
      continue;
    }
    tasks.push(
      refreshLayer(input.dataset.layer).then((meta) => {
        status.push(`${input.dataset.layer}${meta?.truncated ? " (truncated)" : ""}`);
      }),
    );
  }
  await Promise.all(tasks);
  status.push(...refreshVisibleComposites());
  document.getElementById("layer-status").textContent = status.length
    ? `Loaded: ${status.join(", ")}`
    : "No overlay layers loaded.";
}

map.on("load", async () => {
  await loadSummary();
  await loadGasModelInfo();
  await loadTopProspects();
  await loadPlanetStatus();
  await loadCompositeCatalog();
  await refreshVisibleLayers();
});

map.on("moveend", () => {
  refreshVisibleLayers().catch(console.error);
});

document.querySelectorAll('input[name="base-layer"]').forEach((input) => {
  input.addEventListener("change", () => {
    setPlanetBase(input.value === "planet");
  });
});

document.getElementById("planet-mosaics").addEventListener("change", () => {
  const current = document.querySelector('input[name="base-layer"]:checked')?.value;
  setPlanetBase(current === "planet");
});

document.querySelectorAll("input[data-layer]").forEach((input) => {
  input.addEventListener("change", () => {
    refreshVisibleLayers().catch(console.error);
  });
});

document.querySelectorAll("input[data-composite]").forEach((input) => {
  input.addEventListener("change", () => {
    refreshVisibleLayers().catch(console.error);
  });
});
