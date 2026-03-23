from __future__ import annotations

import json
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import rasterio
import requests
from pyproj import Transformer
from pystac import Item
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.transform import Affine
from rasterio.transform import from_bounds
from rasterio.windows import Window
from rasterio.warp import reproject


DEFAULT_STAC_API_URL = "https://stac.dataspace.copernicus.eu/v1"
DEFAULT_PIXEL_ACCESS_MODE = "direct"
DEFAULT_CDSE_PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"
DEFAULT_CDSE_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
)
DEFAULT_CDSE_S3_ENDPOINT_URL = "https://eodata.dataspace.copernicus.eu"
DEFAULT_CDSE_S3_REGION = "default"
DEFAULT_CDSE_RESOLUTION_M = 10.0
DEFAULT_CDSE_REQUEST_TIMEOUT = 120
DEFAULT_S2_BAND_ASSETS = (
    "B02_10m",
    "B03_10m",
    "B04_10m",
    "B05_20m",
    "B06_20m",
    "B07_20m",
    "B08_10m",
    "B8A_20m",
    "B11_20m",
    "B12_20m",
)
DEFAULT_S1_BAND_ASSETS = ("VV", "VH")
DEFAULT_S2_SCL_ASSET = "SCL_20m"
DEFAULT_S2_INVALID_SCL = (0, 1, 3, 8, 9, 10, 11)
CDSE_CLIENT_ID_ENV_KEYS = ("CDSE_CLIENT_ID", "SENTINELHUB_CLIENT_ID", "SH_CLIENT_ID")
CDSE_CLIENT_SECRET_ENV_KEYS = ("CDSE_CLIENT_SECRET", "SENTINELHUB_CLIENT_SECRET", "SH_CLIENT_SECRET")
CDSE_S3_ACCESS_KEY_ENV_KEYS = ("CDSE_S3_ACCESS_KEY", "AWS_ACCESS_KEY_ID")
CDSE_S3_SECRET_KEY_ENV_KEYS = ("CDSE_S3_SECRET_KEY", "AWS_SECRET_ACCESS_KEY")
DEFAULT_DIRECT_ASSET_SOURCE = "remote"

_TOKEN_CACHE: dict[tuple[str, str], dict[str, Any]] = {}
_TRANSFORMER_CACHE: dict[tuple[str, str], Transformer] = {}
_RASTER_DATASET_CACHE: "OrderedDict[str, rasterio.io.DatasetReader]" = OrderedDict()
_CDSE_S3_CLIENT_CACHE: dict[tuple[str, str, str], Any] = {}
DEFAULT_MAX_OPEN_RASTERS = 64
_DIRECT_ASSET_CONFIG: dict[str, Any] = {
    "source": DEFAULT_DIRECT_ASSET_SOURCE,
    "cache_dir": None,
    "cdse_s3_endpoint_url": DEFAULT_CDSE_S3_ENDPOINT_URL,
    "cdse_s3_region": DEFAULT_CDSE_S3_REGION,
}


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported table extension for {path}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def sanitize_id(value: str) -> str:
    return str(value).replace("::", "__").replace("/", "_").replace(":", "_")


def parse_asset_list(text: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(text, (list, tuple)):
        return [str(value).strip() for value in text if str(value).strip()]
    return [part.strip() for part in str(text).split(",") if part.strip()]


def parse_invalid_scl_values(text: str | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(text, (list, tuple)):
        return [int(value) for value in text]
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def parse_iso_day_of_year(value: str | None) -> int:
    if value is None or value == "":
        return -1
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        return -1
    return int(timestamp.dayofyear)


def resolve_asset(item: Item, asset_key: str):
    exact = item.assets.get(asset_key)
    if exact is not None:
        return exact
    target = str(asset_key).lower()
    for key, asset in item.assets.items():
        if str(key).lower() == target:
            return asset
    raise KeyError(asset_key)


def asset_key_to_process_band(asset_key: str) -> str:
    key = str(asset_key).strip()
    if "_" in key:
        prefix, suffix = key.rsplit("_", 1)
        if suffix.endswith("m"):
            return prefix
    return key


def epsg_uri(epsg_code: int) -> str:
    return f"http://www.opengis.net/def/crs/EPSG/0/{int(epsg_code)}"


def local_utm_epsg(lon: float, lat: float) -> int:
    zone = int((float(lon) + 180.0) // 6.0) + 1
    return (32600 if float(lat) >= 0.0 else 32700) + zone


def _max_open_rasters() -> int:
    raw = os.environ.get("EWM_MAX_OPEN_RASTERS", str(DEFAULT_MAX_OPEN_RASTERS))
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return DEFAULT_MAX_OPEN_RASTERS


def _cached_transformer(src_crs: Any, dst_crs: Any) -> Transformer:
    cache_key = (str(src_crs), str(dst_crs))
    cached = _TRANSFORMER_CACHE.get(cache_key)
    if cached is not None:
        return cached
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    _TRANSFORMER_CACHE[cache_key] = transformer
    return transformer


def _get_open_raster(asset_href: str):
    resolved_href = resolve_direct_asset_href(asset_href)
    cached = _RASTER_DATASET_CACHE.get(resolved_href)
    if cached is not None:
        _RASTER_DATASET_CACHE.move_to_end(resolved_href)
        return cached

    dataset = rasterio.open(resolved_href)
    _RASTER_DATASET_CACHE[resolved_href] = dataset

    while len(_RASTER_DATASET_CACHE) > _max_open_rasters():
        _old_href, old_dataset = _RASTER_DATASET_CACHE.popitem(last=False)
        try:
            old_dataset.close()
        except Exception:
            pass

    return dataset


def projected_chip_grid(
    *,
    lon: float,
    lat: float,
    chip_size: int,
    resolution_m: float,
) -> tuple[int, list[float], Affine]:
    epsg_code = local_utm_epsg(lon, lat)
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_code}", always_xy=True)
    x, y = transformer.transform(float(lon), float(lat))
    half_extent = float(chip_size) * float(resolution_m) / 2.0
    bbox = [x - half_extent, y - half_extent, x + half_extent, y + half_extent]
    transform = from_bounds(*bbox, width=int(chip_size), height=int(chip_size))
    return epsg_code, bbox, transform


def fixed_chip_target_grid(
    *,
    lon: float,
    lat: float,
    chip_size: int,
    resolution_m: float,
) -> tuple[str, Affine]:
    epsg_code, _bbox, transform = projected_chip_grid(
        lon=lon,
        lat=lat,
        chip_size=chip_size,
        resolution_m=resolution_m,
    )
    return f"EPSG:{int(epsg_code)}", transform


def process_time_range(
    item: Item,
    *,
    minutes_pad: int = 5,
    mode: str = "around_item",
) -> dict[str, str]:
    if item.datetime is None:
        raise ValueError(f"STAC item {item.id} is missing datetime")
    timestamp = pd.Timestamp(item.datetime)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "day":
        start = timestamp.floor("D").isoformat().replace("+00:00", "Z")
        end = (timestamp.floor("D") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)).isoformat().replace(
            "+00:00",
            "Z",
        )
        return {"from": start, "to": end}
    if normalized_mode != "around_item":
        raise ValueError(f"Unsupported process time range mode: {mode}")
    start = (timestamp - pd.Timedelta(minutes=int(minutes_pad))).isoformat().replace("+00:00", "Z")
    end = (timestamp + pd.Timedelta(minutes=int(minutes_pad))).isoformat().replace("+00:00", "Z")
    return {"from": start, "to": end}


def read_env_first(keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = os.environ.get(key)
        if value:
            return value
    return None


def require_cdse_s3_credentials() -> tuple[str, str]:
    access_key = read_env_first(CDSE_S3_ACCESS_KEY_ENV_KEYS)
    secret_key = read_env_first(CDSE_S3_SECRET_KEY_ENV_KEYS)
    if access_key and secret_key:
        return access_key, secret_key
    raise RuntimeError(
        "CDSE S3 credentials not found. Set one of "
        f"{CDSE_S3_ACCESS_KEY_ENV_KEYS} and one of {CDSE_S3_SECRET_KEY_ENV_KEYS} to use direct_asset_source=cdse_s3_cache."
    )


def configure_direct_asset_access(
    *,
    source: str,
    cache_dir: str | os.PathLike[str] | None,
    cdse_s3_endpoint_url: str = DEFAULT_CDSE_S3_ENDPOINT_URL,
    cdse_s3_region: str = DEFAULT_CDSE_S3_REGION,
) -> None:
    normalized_source = str(source or DEFAULT_DIRECT_ASSET_SOURCE).strip().lower()
    if normalized_source not in {"remote", "cdse_s3_cache"}:
        raise ValueError(f"Unsupported direct asset source: {source}")
    _DIRECT_ASSET_CONFIG["source"] = normalized_source
    _DIRECT_ASSET_CONFIG["cache_dir"] = str(Path(cache_dir).expanduser().resolve()) if cache_dir else None
    _DIRECT_ASSET_CONFIG["cdse_s3_endpoint_url"] = str(cdse_s3_endpoint_url)
    _DIRECT_ASSET_CONFIG["cdse_s3_region"] = str(cdse_s3_region)
    if normalized_source == "cdse_s3_cache" and _DIRECT_ASSET_CONFIG["cache_dir"]:
        Path(_DIRECT_ASSET_CONFIG["cache_dir"]).mkdir(parents=True, exist_ok=True)


def _cdse_s3_client():
    access_key, secret_key = require_cdse_s3_credentials()
    endpoint_url = str(_DIRECT_ASSET_CONFIG.get("cdse_s3_endpoint_url") or DEFAULT_CDSE_S3_ENDPOINT_URL)
    region_name = str(_DIRECT_ASSET_CONFIG.get("cdse_s3_region") or DEFAULT_CDSE_S3_REGION)
    cache_key = (endpoint_url, region_name, access_key)
    cached = _CDSE_S3_CLIENT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:
        raise RuntimeError(
            "boto3 is required for direct_asset_source=cdse_s3_cache. Install boto3 to use CDSE S3 cached direct reads."
        ) from exc
    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region_name,
        config=Config(
            signature_version="s3v4",
            retries={"max_attempts": 5, "mode": "standard"},
            s3={"addressing_style": "path"},
        ),
    )
    _CDSE_S3_CLIENT_CACHE[cache_key] = client
    return client


def parse_cdse_s3_location(asset_href: str) -> tuple[str, str] | None:
    text = str(asset_href).strip()
    if not text:
        return None
    if text.startswith("s3://"):
        parsed = urlparse(text)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        return (bucket, key) if bucket and key else None

    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"}:
        return None
    host = (parsed.netloc or "").lower()
    if host != "eodata.dataspace.copernicus.eu":
        return None
    path = parsed.path.lstrip("/")
    if not path:
        return None
    parts = path.split("/", 1)
    if len(parts) != 2:
        return None
    bucket, key = parts
    if not bucket or not key:
        return None
    return bucket, key


def _local_cache_path_for_s3_object(bucket: str, key: str) -> Path:
    cache_dir = _DIRECT_ASSET_CONFIG.get("cache_dir")
    if not cache_dir:
        raise RuntimeError(
            "direct_asset_source=cdse_s3_cache requires a local cache dir. Configure local_asset_cache_dir before materialization."
        )
    cache_root = Path(str(cache_dir))
    return cache_root / bucket / Path(key)


def _download_cdse_s3_object(bucket: str, key: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination

    client = _cdse_s3_client()
    tmp_path = destination.with_name(f"{destination.name}.part.{os.getpid()}.{int(time.time() * 1000)}")
    try:
        client.download_file(bucket, key, str(tmp_path))
        tmp_path.replace(destination)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
    return destination


def resolve_direct_asset_href(asset_href: str) -> str:
    source = str(_DIRECT_ASSET_CONFIG.get("source") or DEFAULT_DIRECT_ASSET_SOURCE).strip().lower()
    if source != "cdse_s3_cache":
        return str(asset_href)

    parsed = parse_cdse_s3_location(asset_href)
    if parsed is None:
        return str(asset_href)
    bucket, key = parsed
    local_path = _local_cache_path_for_s3_object(bucket, key)
    cached_path = _download_cdse_s3_object(bucket, key, local_path)
    return str(cached_path)


def require_cdse_client_credentials() -> tuple[str, str]:
    client_id = read_env_first(CDSE_CLIENT_ID_ENV_KEYS)
    client_secret = read_env_first(CDSE_CLIENT_SECRET_ENV_KEYS)
    if client_id and client_secret:
        return client_id, client_secret
    raise RuntimeError(
        "CDSE/Sentinel Hub client credentials not found. Set one of "
        f"{CDSE_CLIENT_ID_ENV_KEYS} and one of {CDSE_CLIENT_SECRET_ENV_KEYS} to use pixel_access_mode=cdse_process."
    )


def fetch_cdse_access_token(
    *,
    token_url: str,
    timeout: int,
) -> str:
    client_id, client_secret = require_cdse_client_credentials()
    cache_key = (str(token_url), str(client_id))
    cached = _TOKEN_CACHE.get(cache_key)
    now = time.time()
    if cached is not None and now < float(cached["expires_at"]) - 60.0:
        return str(cached["access_token"])

    response = requests.post(
        str(token_url),
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=int(timeout),
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"Failed to fetch CDSE token status={response.status_code}: {response.text[:400]}"
        )
    payload = response.json()
    access_token = payload.get("access_token")
    if not access_token:
        raise RuntimeError("CDSE token response did not include access_token")
    expires_in = int(payload.get("expires_in", 3600))
    _TOKEN_CACHE[cache_key] = {
        "access_token": access_token,
        "expires_at": now + expires_in,
    }
    return str(access_token)


def post_cdse_process_request(
    *,
    request_payload: dict[str, Any],
    process_url: str,
    token_url: str,
    timeout: int,
) -> bytes:
    access_token = fetch_cdse_access_token(token_url=token_url, timeout=timeout)
    response = requests.post(
        str(process_url),
        json=request_payload,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Accept": "image/tiff",
        },
        timeout=int(timeout),
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"CDSE process request failed status={response.status_code}: {response.text[:400]}"
        )
    return response.content


def read_tiff_from_bytes(content: bytes) -> tuple[np.ndarray, Any, Affine]:
    with MemoryFile(content) as memfile:
        with memfile.open() as ds:
            array = ds.read().astype(np.float32)
            return array, ds.crs, ds.transform


def build_s2_process_evalscript(
    *,
    band_assets: list[str],
    scl_asset: str | None,
) -> str:
    output_bands = [asset_key_to_process_band(asset) for asset in band_assets]
    if scl_asset:
        output_bands.append(asset_key_to_process_band(scl_asset))
    inputs = ", ".join(f'"{band}"' for band in (output_bands + ["dataMask"]))
    outputs = ", ".join(f"sample.{band}" for band in (output_bands + ["dataMask"]))
    return f"""//VERSION=3
function setup() {{
  return {{
    input: [{inputs}],
    output: {{
      bands: {len(output_bands) + 1},
      sampleType: "FLOAT32"
    }}
  }};
}}

function evaluatePixel(sample) {{
  return [{outputs}];
}}
"""


def build_s1_process_evalscript(*, band_assets: list[str]) -> str:
    output_bands = [asset_key_to_process_band(asset) for asset in band_assets]
    input_bands = output_bands + ["dataMask"]
    inputs = ", ".join(f'"{band}"' for band in input_bands)
    outputs = ", ".join(f"sample.{band}" for band in input_bands)
    return f"""//VERSION=3
function setup() {{
  return {{
    input: [{inputs}],
    output: {{
      bands: {len(input_bands)},
      sampleType: "FLOAT32"
    }}
  }};
}}

function evaluatePixel(sample) {{
  return [{outputs}];
}}
"""


def build_s2s1_fusion_process_evalscript(
    *,
    s2_band_assets: list[str],
    s2_scl_asset: str | None,
    s1_band_assets: list[str],
) -> str:
    s2_output_bands = [asset_key_to_process_band(asset) for asset in s2_band_assets]
    if s2_scl_asset:
        s2_output_bands.append(asset_key_to_process_band(s2_scl_asset))
    s1_output_bands = [asset_key_to_process_band(asset) for asset in s1_band_assets]

    total_bands = len(s2_output_bands) + 1 + len(s1_output_bands) + 1
    s2_outputs = ", ".join(f"sampleOrZero(s2, '{band}')" for band in s2_output_bands)
    s1_outputs = ", ".join(f"sampleOrZero(s1, '{band}')" for band in s1_output_bands)
    return f"""//VERSION=3
function setup() {{
  return {{
    input: [
      {{
        datasource: "s2",
        bands: [{", ".join(f'"{band}"' for band in (s2_output_bands + ["dataMask"]))}]
      }},
      {{
        datasource: "s1",
        bands: [{", ".join(f'"{band}"' for band in (s1_output_bands + ["dataMask"]))}]
      }}
    ],
    output: {{
      bands: {total_bands},
      sampleType: "FLOAT32"
    }}
  }};
}}

function sampleOrZero(sample, key) {{
  if (!sample || sample[key] === undefined || sample[key] === null) {{
    return 0;
  }}
  return sample[key];
}}

function evaluatePixel(samples) {{
  let s2 = (samples.s2 && samples.s2.length > 0) ? samples.s2[0] : null;
  let s1 = (samples.s1 && samples.s1.length > 0) ? samples.s1[0] : null;
  return [{s2_outputs}, sampleOrZero(s2, 'dataMask'), {s1_outputs}, sampleOrZero(s1, 'dataMask')];
}}
"""


def build_process_request(
    *,
    collection_type: str,
    item: Item,
    lon: float,
    lat: float,
    chip_size: int,
    resolution_m: float,
    evalscript: str,
    data_filter: dict[str, Any] | None = None,
    processing: dict[str, Any] | None = None,
    time_range_mode: str = "around_item",
) -> tuple[dict[str, Any], int]:
    epsg_code, bbox, _transform = projected_chip_grid(
        lon=lon,
        lat=lat,
        chip_size=chip_size,
        resolution_m=resolution_m,
    )
    data_spec: dict[str, Any] = {
        "type": str(collection_type),
        "dataFilter": {
            "timeRange": process_time_range(item, mode=time_range_mode),
        },
    }
    if data_filter:
        data_spec["dataFilter"].update(data_filter)
    if processing:
        data_spec["processing"] = processing
    request_payload = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {"crs": epsg_uri(epsg_code)},
            },
            "data": [data_spec],
        },
        "output": {
            "width": int(chip_size),
            "height": int(chip_size),
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": evalscript,
    }
    return request_payload, epsg_code


def build_s2s1_fusion_process_request(
    *,
    s2_item: Item,
    s1_item: Item,
    lon: float,
    lat: float,
    chip_size: int,
    resolution_m: float,
    evalscript: str,
) -> tuple[dict[str, Any], int]:
    epsg_code, bbox, _transform = projected_chip_grid(
        lon=lon,
        lat=lat,
        chip_size=chip_size,
        resolution_m=resolution_m,
    )
    request_payload = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {"crs": epsg_uri(epsg_code)},
            },
            "data": [
                {
                    "id": "s2",
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": process_time_range(s2_item, mode="day"),
                    },
                },
                {
                    "id": "s1",
                    "type": "sentinel-1-grd",
                    "dataFilter": {
                        "timeRange": process_time_range(s1_item, mode="around_item"),
                        "resolution": "HIGH",
                    },
                    "processing": {
                        "orthorectify": "true",
                        "backCoeff": "GAMMA0_TERRAIN",
                        "demInstance": "COPERNICUS_30",
                    },
                },
            ],
        },
        "output": {
            "width": int(chip_size),
            "height": int(chip_size),
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": evalscript,
    }
    return request_payload, epsg_code


def assign_sequence_shard(
    sequences: pd.DataFrame,
    *,
    shard_index: int,
    shard_count: int,
) -> pd.DataFrame:
    if shard_count <= 1:
        return sequences.copy().reset_index(drop=True)
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"shard_index must be in [0, {shard_count}), got {shard_index}")
    ordered = sequences.sort_values("sequence_id").reset_index(drop=True).copy()
    return ordered.iloc[shard_index::shard_count].copy().reset_index(drop=True)


def apply_signing(item: Item, signing_mode: str) -> Item:
    normalized = str(signing_mode).strip().lower()
    if normalized in {"", "none"}:
        return item
    if normalized == "planetary_computer":
        import planetary_computer

        return planetary_computer.sign(item)
    raise ValueError(f"Unsupported signing_mode: {signing_mode}")


def fetch_item_by_id(
    catalog: Client,
    *,
    collection: str,
    item_id: str,
    signing_mode: str,
    cache: dict[tuple[str, str], Item],
) -> Item:
    cache_key = (str(collection), str(item_id))
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    items = list(catalog.search(collections=[collection], ids=[item_id], max_items=1).items())
    if not items:
        raise RuntimeError(f"Could not resolve STAC item collection={collection} item_id={item_id}")
    signed = apply_signing(items[0], signing_mode=signing_mode)
    cache[cache_key] = signed
    return signed


def reference_window_and_transform(
    asset_href: str,
    *,
    lon: float,
    lat: float,
    chip_size: int,
) -> tuple[Any, Window, Affine]:
    with rasterio.Env(AWS_NO_SIGN_REQUEST="YES"):
        ds = _get_open_raster(asset_href)
        transformer = _cached_transformer("EPSG:4326", ds.crs)
        x, y = transformer.transform(float(lon), float(lat))
        row, col = ds.index(x, y)
        window = Window(col - (chip_size // 2), row - (chip_size // 2), chip_size, chip_size)
        return ds.crs, window, ds.window_transform(window)


def reproject_asset_to_grid(
    asset_href: str,
    *,
    dst_crs: Any,
    dst_transform: Affine,
    width: int,
    height: int,
    resampling: Resampling,
    dst_nodata: float = np.nan,
) -> np.ndarray:
    destination = np.full((height, width), dst_nodata, dtype=np.float32)
    with rasterio.Env(AWS_NO_SIGN_REQUEST="YES"):
        src = _get_open_raster(asset_href)
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=dst_nodata,
            resampling=resampling,
        )
    return destination


def empty_frame(
    *,
    band_count: int,
    chip_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.full((band_count, chip_size, chip_size), np.nan, dtype=np.float32),
        np.zeros((chip_size, chip_size), dtype=bool),
    )


def read_s2_frame(
    item: Item,
    *,
    lon: float,
    lat: float,
    chip_size: int,
    band_assets: list[str],
    scl_asset: str | None,
    invalid_scl_values: list[int],
    dst_crs: Any | None = None,
    dst_transform: Affine | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], Any, Affine]:
    if dst_crs is None or dst_transform is None:
        reference_asset = band_assets[0]
        reference = resolve_asset(item, reference_asset)
        dst_crs, _window, dst_transform = reference_window_and_transform(
            reference.href,
            lon=lon,
            lat=lat,
            chip_size=chip_size,
        )

    valid_mask: np.ndarray
    clear_fraction: float | None = None
    if scl_asset:
        scl = resolve_asset(item, scl_asset)
        scl = reproject_asset_to_grid(
            scl.href,
            dst_crs=dst_crs,
            dst_transform=dst_transform,
            width=chip_size,
            height=chip_size,
            resampling=Resampling.nearest,
        )
        scl_int = np.where(np.isfinite(scl), np.rint(scl), -9999).astype(np.int16)
        valid_mask = np.isfinite(scl) & ~np.isin(scl_int, invalid_scl_values)
        clear_fraction = float(valid_mask.mean())
    else:
        valid_mask = np.ones((chip_size, chip_size), dtype=bool)

    bands: list[np.ndarray] = []
    for asset_key in band_assets:
        asset = resolve_asset(item, asset_key)
        band = reproject_asset_to_grid(
            asset.href,
            dst_crs=dst_crs,
            dst_transform=dst_transform,
            width=chip_size,
            height=chip_size,
            resampling=Resampling.bilinear,
        )
        band_valid = np.isfinite(band) & valid_mask
        bands.append(np.where(band_valid, band, np.nan).astype(np.float32))
        valid_mask = valid_mask & np.isfinite(band)

    metadata = {
        "item_id": item.id,
        "datetime": item.datetime.isoformat() if item.datetime is not None else None,
        "cloud_cover": item.properties.get("eo:cloud_cover"),
        "clear_fraction": clear_fraction,
    }
    return np.stack(bands, axis=0), valid_mask.astype(bool), metadata, dst_crs, dst_transform


def evaluate_s2_chip_quality(
    item: Item,
    *,
    lon: float,
    lat: float,
    chip_size: int,
    scl_asset: str | None,
    invalid_scl_values: list[int],
    dst_crs: Any | None = None,
    dst_transform: Affine | None = None,
    reference_asset_key: str = "B02",
) -> tuple[np.ndarray, dict[str, Any], Any, Affine]:
    if dst_crs is None or dst_transform is None:
        reference = resolve_asset(item, reference_asset_key)
        dst_crs, _window, dst_transform = reference_window_and_transform(
            reference.href,
            lon=lon,
            lat=lat,
            chip_size=chip_size,
        )

    if scl_asset:
        scl = resolve_asset(item, scl_asset)
        scl = reproject_asset_to_grid(
            scl.href,
            dst_crs=dst_crs,
            dst_transform=dst_transform,
            width=chip_size,
            height=chip_size,
            resampling=Resampling.nearest,
        )
        scl_int = np.where(np.isfinite(scl), np.rint(scl), -9999).astype(np.int16)
        footprint_mask = np.isfinite(scl) & ~np.isin(scl_int, [0, 1])
        valid_mask = footprint_mask & ~np.isin(scl_int, invalid_scl_values)
    else:
        footprint_mask = np.ones((chip_size, chip_size), dtype=bool)
        valid_mask = np.ones((chip_size, chip_size), dtype=bool)

    footprint_count = int(footprint_mask.sum())
    clear_fraction = float(valid_mask.sum() / footprint_count) if footprint_count > 0 else 0.0
    metadata = {
        "item_id": item.id,
        "datetime": item.datetime.isoformat() if item.datetime is not None else None,
        "cloud_cover": item.properties.get("eo:cloud_cover"),
        "clear_fraction": clear_fraction,
        "valid_fraction": float(valid_mask.mean()) if valid_mask.size > 0 else 0.0,
        "footprint_fraction": float(footprint_mask.mean()) if footprint_mask.size > 0 else 0.0,
    }
    return valid_mask.astype(bool), metadata, dst_crs, dst_transform


def read_s2_frame_process(
    item: Item,
    *,
    lon: float,
    lat: float,
    chip_size: int,
    band_assets: list[str],
    scl_asset: str | None,
    invalid_scl_values: list[int],
    process_url: str,
    token_url: str,
    resolution_m: float,
    request_timeout: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], Any, Affine]:
    request_payload, epsg_code = build_process_request(
        collection_type="sentinel-2-l2a",
        item=item,
        lon=lon,
        lat=lat,
        chip_size=chip_size,
        resolution_m=resolution_m,
        evalscript=build_s2_process_evalscript(band_assets=band_assets, scl_asset=scl_asset),
        time_range_mode="day",
    )
    content = post_cdse_process_request(
        request_payload=request_payload,
        process_url=process_url,
        token_url=token_url,
        timeout=request_timeout,
    )
    array, dst_crs, dst_transform = read_tiff_from_bytes(content)
    expected_band_count = len(band_assets) + (1 if scl_asset else 0) + 1
    if array.shape[0] != expected_band_count:
        raise RuntimeError(
            f"Unexpected S2 process response bands={array.shape[0]} expected={expected_band_count}"
        )

    data_mask = array[-1] > 0.5
    valid_mask = np.ones((chip_size, chip_size), dtype=bool)
    clear_fraction: float | None = None
    if scl_asset:
        scl = array[-2]
        scl_int = np.where(np.isfinite(scl), np.rint(scl), -9999).astype(np.int16)
        valid_mask = data_mask & np.isfinite(scl) & ~np.isin(scl_int, invalid_scl_values)
        clear_fraction = float(valid_mask.mean())
        band_stack = array[:-2]
    else:
        valid_mask = data_mask.copy()
        band_stack = array[:-1]

    bands: list[np.ndarray] = []
    for band in band_stack:
        band_valid = np.isfinite(band) & valid_mask
        bands.append(np.where(band_valid, band, np.nan).astype(np.float32))
        valid_mask = valid_mask & np.isfinite(band)

    metadata = {
        "item_id": item.id,
        "datetime": item.datetime.isoformat() if item.datetime is not None else None,
        "cloud_cover": item.properties.get("eo:cloud_cover"),
        "clear_fraction": clear_fraction,
        "pixel_access_mode": "cdse_process",
        "process_epsg": int(epsg_code),
    }
    return np.stack(bands, axis=0), valid_mask.astype(bool), metadata, dst_crs, dst_transform


def read_s1_frame(
    item: Item,
    *,
    dst_crs: Any,
    dst_transform: Affine,
    chip_size: int,
    band_assets: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    valid_mask = np.ones((chip_size, chip_size), dtype=bool)
    bands: list[np.ndarray] = []

    for asset_key in band_assets:
        asset = resolve_asset(item, asset_key)
        band = reproject_asset_to_grid(
            asset.href,
            dst_crs=dst_crs,
            dst_transform=dst_transform,
            width=chip_size,
            height=chip_size,
            resampling=Resampling.bilinear,
        )
        valid_mask = valid_mask & np.isfinite(band)
        bands.append(np.where(np.isfinite(band), band, np.nan).astype(np.float32))

    metadata = {
        "item_id": item.id,
        "datetime": item.datetime.isoformat() if item.datetime is not None else None,
        "valid_fraction": float(valid_mask.mean()),
    }
    return np.stack(bands, axis=0), valid_mask.astype(bool), metadata


def read_s1_frame_process(
    item: Item,
    *,
    lon: float,
    lat: float,
    chip_size: int,
    band_assets: list[str],
    process_url: str,
    token_url: str,
    resolution_m: float,
    request_timeout: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    request_payload, epsg_code = build_process_request(
        collection_type="sentinel-1-grd",
        item=item,
        lon=lon,
        lat=lat,
        chip_size=chip_size,
        resolution_m=resolution_m,
        evalscript=build_s1_process_evalscript(band_assets=band_assets),
        data_filter={"resolution": "HIGH"},
        processing={
            "orthorectify": "true",
            "backCoeff": "GAMMA0_TERRAIN",
            "demInstance": "COPERNICUS_30",
        },
    )
    content = post_cdse_process_request(
        request_payload=request_payload,
        process_url=process_url,
        token_url=token_url,
        timeout=request_timeout,
    )
    array, _dst_crs, _dst_transform = read_tiff_from_bytes(content)
    expected_band_count = len(band_assets) + 1
    if array.shape[0] != expected_band_count:
        raise RuntimeError(
            f"Unexpected S1 process response bands={array.shape[0]} expected={expected_band_count}"
        )
    band_stack = array[:-1]
    data_mask = array[-1] > 0.5

    valid_mask = data_mask.astype(bool)
    bands: list[np.ndarray] = []
    for band in band_stack:
        valid_mask = valid_mask & np.isfinite(band)
        bands.append(np.where(np.isfinite(band) & data_mask, band, np.nan).astype(np.float32))

    metadata = {
        "item_id": item.id,
        "datetime": item.datetime.isoformat() if item.datetime is not None else None,
        "valid_fraction": float(valid_mask.mean()),
        "pixel_access_mode": "cdse_process",
        "process_epsg": int(epsg_code),
    }
    return np.stack(bands, axis=0), valid_mask.astype(bool), metadata


def read_s2s1_frame_process_fused(
    s2_item: Item,
    s1_item: Item,
    *,
    lon: float,
    lat: float,
    chip_size: int,
    s2_band_assets: list[str],
    s2_scl_asset: str | None,
    invalid_scl_values: list[int],
    s1_band_assets: list[str],
    process_url: str,
    token_url: str,
    resolution_m: float,
    request_timeout: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], np.ndarray, np.ndarray, dict[str, Any], Any, Affine]:
    request_payload, epsg_code = build_s2s1_fusion_process_request(
        s2_item=s2_item,
        s1_item=s1_item,
        lon=lon,
        lat=lat,
        chip_size=chip_size,
        resolution_m=resolution_m,
        evalscript=build_s2s1_fusion_process_evalscript(
            s2_band_assets=s2_band_assets,
            s2_scl_asset=s2_scl_asset,
            s1_band_assets=s1_band_assets,
        ),
    )
    content = post_cdse_process_request(
        request_payload=request_payload,
        process_url=process_url,
        token_url=token_url,
        timeout=request_timeout,
    )
    array, dst_crs, dst_transform = read_tiff_from_bytes(content)

    s2_extra = (1 if s2_scl_asset else 0) + 1
    expected_band_count = len(s2_band_assets) + s2_extra + len(s1_band_assets) + 1
    if array.shape[0] != expected_band_count:
        raise RuntimeError(
            f"Unexpected fused process response bands={array.shape[0]} expected={expected_band_count}"
        )

    cursor = 0
    s2_band_stack = array[cursor : cursor + len(s2_band_assets)]
    cursor += len(s2_band_assets)
    s2_scl = array[cursor] if s2_scl_asset else None
    if s2_scl_asset:
        cursor += 1
    s2_data_mask = array[cursor] > 0.5
    cursor += 1
    s1_band_stack = array[cursor : cursor + len(s1_band_assets)]
    cursor += len(s1_band_assets)
    s1_data_mask = array[cursor] > 0.5

    if s2_scl is not None:
        s2_scl_int = np.where(np.isfinite(s2_scl), np.rint(s2_scl), -9999).astype(np.int16)
        s2_valid_mask = s2_data_mask & np.isfinite(s2_scl) & ~np.isin(s2_scl_int, invalid_scl_values)
        s2_clear_fraction = float(s2_valid_mask.mean())
    else:
        s2_valid_mask = s2_data_mask.copy()
        s2_clear_fraction = float(s2_valid_mask.mean())

    s2_bands: list[np.ndarray] = []
    for band in s2_band_stack:
        band_valid = np.isfinite(band) & s2_valid_mask
        s2_bands.append(np.where(band_valid, band, np.nan).astype(np.float32))
        s2_valid_mask = s2_valid_mask & np.isfinite(band)

    s1_valid_mask = s1_data_mask.astype(bool)
    s1_bands: list[np.ndarray] = []
    for band in s1_band_stack:
        s1_valid_mask = s1_valid_mask & np.isfinite(band)
        s1_bands.append(np.where(np.isfinite(band) & s1_data_mask, band, np.nan).astype(np.float32))

    s2_metadata = {
        "item_id": s2_item.id,
        "datetime": s2_item.datetime.isoformat() if s2_item.datetime is not None else None,
        "cloud_cover": s2_item.properties.get("eo:cloud_cover"),
        "clear_fraction": s2_clear_fraction,
        "pixel_access_mode": "cdse_process_fused",
        "process_epsg": int(epsg_code),
    }
    s1_metadata = {
        "item_id": s1_item.id,
        "datetime": s1_item.datetime.isoformat() if s1_item.datetime is not None else None,
        "valid_fraction": float(s1_valid_mask.mean()),
        "pixel_access_mode": "cdse_process_fused",
        "process_epsg": int(epsg_code),
    }
    return (
        np.stack(s2_bands, axis=0),
        s2_valid_mask.astype(bool),
        s2_metadata,
        np.stack(s1_bands, axis=0),
        s1_valid_mask.astype(bool),
        s1_metadata,
        dst_crs,
        dst_transform,
    )
