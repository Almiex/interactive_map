# -*- coding: utf-8 -*-
"""
GeoHex Analytics — Streamlit-сервис аналитики города по гексагональной сетке Uber H3.

Источники (все бесплатные):
  * Геокодинг городов        — Nominatim (OSM)
  * Здания / дороги / POI    — Overpass API (OpenStreetMap)
  * Реальная численность     — Kontur Population (локальный файл, https://data.humdata.org)

Запуск:  streamlit run app.py
"""

import time
import requests
import numpy as np
import pandas as pd
import h3
import folium
import streamlit as st
import geopandas as gpd
from shapely.geometry import Polygon
from streamlit_folium import st_folium
from branca.colormap import LinearColormap

# --------------------------------------------------------------------------- #
#  Константы
# --------------------------------------------------------------------------- #
OVERPASS_ENDPOINTS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
HEADERS = {"User-Agent": "GeoHexAnalytics/1.0 (educational; OSM data)"}

# расстояние между центрами соседних гексов, км (по стандарту Uber H3)
RES_SPACING_KM = {7: 2.4, 8: 0.92, 9: 0.35, 10: 0.13}

MAX_GRID_CELLS = 25000  # потолок сетки; отрисовка — одним GeoJSON-слоем

MAP_TYPES = [
    "1. Плотность населения",
    "2. Объём жилого фонда",
    "3. Инфраструктура и POI",
    "4. Транспортная доступность",
    "5. Социальная инфраструктура",
    "6. Трафик (пеший / автомобильный)",
    "7. Конкурентная среда (медицина)",
]

POI_CATEGORIES = {
    "Все POI (любые)": None,
    "Продуктовые магазины": ("shop", {"supermarket", "convenience", "greengrocer",
                                      "deli", "bakery", "butcher", "seafood"}),
    "Общепит": ("amenity", {"restaurant", "cafe", "fast_food", "bar", "pub",
                            "food_court", "biergarten"}),
    "Аптеки": ("amenity", {"pharmacy"}),
    "Салоны красоты": ("shop", {"hairdresser", "beauty", "massage", "tattoo"}),
    "Спорт и фитнес": ("leisure", {"fitness_centre", "sports_centre", "pitch",
                                   "stadium", "track", "swimming_pool"}),
    "Банки и банкоматы": ("amenity", {"bank", "atm", "bureau_de_change"}),
    "Образование": ("amenity", {"school", "kindergarten", "college", "university",
                                "library", "driving_school", "language_school"}),
    "Офисы и услуги (office/craft)": "__OFFICE_CRAFT__",
}

SOCIAL_CATEGORIES = {
    "Школы": ("amenity", {"school"}),
    "Детские сады": ("amenity", {"kindergarten"}),
    "Больницы": ("amenity", {"hospital"}),
    "Поликлиники/клиники": ("amenity", {"clinic", "doctors"}),
    "Спортплощадки": ("leisure", {"pitch", "sports_centre", "stadium", "fitness_centre"}),
}

COMPETITOR_TAGS = ("amenity", {"clinic", "doctors", "hospital", "pharmacy"})  # dentist НЕ включаем!

TRANSPORT_MODES = [
    "Остановки общественного транспорта (кол-во)",
    "Парковки (кол-во)",
    "Парковки (площадь, м²)",
    "Плотность дорожной сети (км/км²)",
]

TRAFFIC_MODES = [
    "Автомобильный (primary/secondary/tertiary и выше)",
    "Пешеходный (footway/pedestrian/path и т.п.)",
]

AUTO_HIGHWAYS = {"motorway", "motorway_link", "trunk", "trunk_link", "primary",
                 "primary_link", "secondary", "secondary_link", "tertiary", "tertiary_link"}
PED_HIGHWAYS = {"footway", "pedestrian", "path", "steps", "cycleway", "living_street"}

# --------------------------------------------------------------------------- #
#  Геокодинг
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=86400, show_spinner=False)
def geocode_city(city: str):
    r = requests.get(NOMINATIM_URL, params={
        "q": city, "format": "json", "limit": 1, "accept-language": "ru",
        "polygon_geojson": 1,          # полигон административной границы
        "polygon_threshold": 0.0005,   # упрощение границы (меньше трафик)
    }, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not data:
        return None
    d = data[0]
    return {
        "lat": float(d["lat"]), "lon": float(d["lon"]),
        "display": d["display_name"],
        "bbox": [float(x) for x in d["boundingbox"]],  # [south, north, west, east]
        "geojson": d.get("geojson"),                   # граница города или None
    }

# --------------------------------------------------------------------------- #
#  H3-сетка
# --------------------------------------------------------------------------- #
def _cells_from_polygon(coordinates, res):
    """coordinates — GeoJSON Polygon: [outer, hole1, ...] в порядке (lng, lat)."""
    outer = [(lat, lng) for lng, lat in coordinates[0]]
    if outer[0] != outer[-1]:
        outer.append(outer[0])
    holes = []
    for hole in coordinates[1:]:
        h = [(lat, lng) for lng, lat in hole]
        if h[0] != h[-1]:
            h.append(h[0])
        holes.append(h)
    try:
        poly = h3.LatLngPoly(outer, *holes)          # h3 >= 4.1
    except AttributeError:
        poly = {"type": "Polygon",
                "coordinates": [[[lng, lat] for lat, lng in outer]] +
                                [[[lng, lat] for lat, lng in h] for h in holes]}
    return h3.polygon_to_cells(poly, res)


def _geojson_bounds(geom):
    """south, north, west, east из GeoJSON Polygon/MultiPolygon."""
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    pts = [p for poly in polys for p in poly[0]]
    lngs = [p[0] for p in pts]
    lats = [p[1] for p in pts]
    return min(lats), max(lats), min(lngs), max(lngs)


def make_grid(geo, res):
    geom = geo.get("geojson")
    if geom and geom.get("type") in ("Polygon", "MultiPolygon"):
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        cells = set()
        for poly in polys:
            cells.update(_cells_from_polygon(poly, res))
        if cells:
            return sorted(cells), True
    # fallback: граница не нашлась (лимит Nominatim 0,5 МБ) — старый прямоугольник
    south, north, west, east = geo["bbox"]
    m = 0.02
    ring = [(south - m, west - m), (south - m, east + m),
            (north + m, east + m), (north + m, west - m),
            (south - m, west - m)]
    try:
        poly = h3.LatLngPoly(ring)
    except AttributeError:
        poly = {"type": "Polygon", "coordinates": [[[lng, lat] for lat, lng in ring]]}
    return sorted(h3.polygon_to_cells(poly, res)), False


def _query_overpass(q: str) -> dict:
    last_err = None
    for url in OVERPASS_ENDPOINTS:
        try:
            r = requests.post(url, data={"data": q}, headers=HEADERS, timeout=420)
            if r.status_code == 200:
                return r.json()
            last_err = f"{url}: HTTP {r.status_code}"
        except Exception as e:  # noqa: BLE001
            last_err = f"{url}: {e}"
        time.sleep(2)
    raise RuntimeError(f"Overpass недоступен: {last_err}")

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_city_data(city: str):
    """Две выгрузки Overpass: (1) way-объекты, (2) node-объекты. Возвращает DataFrames."""
    geo = geocode_city(city)
    if geo is None:
        return None, None, None
    if geo.get("geojson") and geo["geojson"].get("type") in ("Polygon", "MultiPolygon"):
        south, north, west, east = _geojson_bounds(geo["geojson"])  # гексы только в городе
    else:
        south, north, west, east = geo["bbox"]
    # запас на границах — чтобы объекты у края гексов не потерялись
    south, north, west, east = south - 0.02, north + 0.02, west - 0.02, east + 0.02
    bb = f"{south},{west},{north},{east}"

    q_ways = f"""
[out:json][timeout:300];
(
  way["building"~"^(apartments|residential)$"]({bb});
  way["highway"]({bb});
  way["amenity"]({bb});
  way["shop"]({bb});
  way["leisure"]({bb});
);
out geom;"""

    q_nodes = f"""
[out:json][timeout:300];
(
  node["amenity"]({bb});
  node["shop"]({bb});
  node["craft"]({bb});
  node["office"]({bb});
  node["leisure"]({bb});
  node["highway"~"^(bus_stop|bus_station|tram_stop)$"]({bb});
  node["public_transport"~"^(platform|stop_position)$"]({bb});
);
out body;"""

    ways_raw = _query_overpass(q_ways).get("elements", [])
    nodes_raw = _query_overpass(q_nodes).get("elements", [])

    ways, nodes = [], []
    for el in ways_raw:
        if "geometry" not in el:
            continue
        ways.append({
            "id": el["id"],
            "coords": [(p["lon"], p["lat"]) for p in el["geometry"]],
            "tags": el.get("tags", {}),
        })
    for el in nodes_raw:
        nodes.append({"id": el["id"], "lat": el["lat"], "lon": el["lon"],
                      "tags": el.get("tags", {})})

    ways_df = pd.DataFrame(ways, columns=["id", "coords", "tags"])
    nodes_df = pd.DataFrame(nodes, columns=["id", "lat", "lon", "tags"])
    return geo, nodes_df, ways_df

# --------------------------------------------------------------------------- #
#  Подготовка слоёв
# --------------------------------------------------------------------------- #
def ways_centroids(ways_df):
    """Центроиды way-объектов (широта/долгота) — для подсчёта POI/остановок-ways."""
    if ways_df.empty:
        return pd.DataFrame(columns=["lat", "lon", "tags"])
    lat = ways_df["coords"].apply(lambda c: np.mean([p[1] for p in c]))
    lon = ways_df["coords"].apply(lambda c: np.mean([p[0] for p in c]))
    return pd.DataFrame({"lat": lat, "lon": lon, "tags": ways_df["tags"]})

def _ring_coords(coords, min_pts):
    """Чистим кольцо/линию от подряд идущих дублей; None если точек < min_pts."""
    pts = []
    for p in coords:
        if not pts or p != pts[-1]:
            pts.append(p)
    return pts if len(pts) >= min_pts else None


def _safe_polygon(coords):
    pts = _ring_coords(coords, 3)
    if pts is None:
        return None
    try:
        poly = Polygon(pts)
        return poly if poly.is_valid else poly.buffer(0)
    except Exception:  # noqa: BLE001
        return None


def buildings_gdf(ways_df):
    """Жилые здания с площадью застройки (м², в метрической проекции) и этажностью."""
    if ways_df.empty:
        return gpd.GeoDataFrame(columns=["levels", "area", "lat", "lon"], geometry=[], crs="EPSG:4326")
    mask = ways_df["tags"].apply(lambda t: t.get("building") in ("apartments", "residential"))
    sub = ways_df[mask]
    if sub.empty:
        return gpd.GeoDataFrame(columns=["levels", "area", "lat", "lon"], geometry=[], crs="EPSG:4326")
    recs, geoms = [], []
    for coords, tags in zip(sub["coords"], sub["tags"]):
        geom = _safe_polygon(coords)
        if geom is None or geom.is_empty:
            continue
        recs.append({"levels": _to_int(tags.get("building:levels"), 1),
                     "lat": float(np.mean([p[1] for p in coords])),
                     "lon": float(np.mean([p[0] for p in coords]))})
        geoms.append(geom)
    if not geoms:
        return gpd.GeoDataFrame(columns=["levels", "area", "lat", "lon"],
                                geometry=[], crs="EPSG:4326")
    gdf = gpd.GeoDataFrame(recs, geometry=geoms, crs="EPSG:4326")
    metric = gdf.to_crs(gdf.estimate_utm_crs())
    gdf["area"] = metric.area.values
    return gdf

def roads_gdf(ways_df):
    """Дороги с длиной (км, в метрической проекции)."""
    if ways_df.empty:
        return gpd.GeoDataFrame(columns=["highway", "len_km", "lat", "lon"], geometry=[], crs="EPSG:4326")
    mask = ways_df["tags"].apply(lambda t: "highway" in t)
    sub = ways_df[mask]
    if sub.empty:
        return gpd.GeoDataFrame(columns=["highway", "len_km", "lat", "lon"], geometry=[], crs="EPSG:4326")
    from shapely.geometry import LineString
    recs, geoms = [], []
    for coords, tags in zip(sub["coords"], sub["tags"]):
        pts = _ring_coords(coords, 2)
        if pts is None:
            continue
        try:
            geom = LineString(pts)
        except Exception:  # noqa: BLE001
            continue
        recs.append({"highway": tags.get("highway", ""),
                     "lat": float(np.mean([p[1] for p in pts])),
                     "lon": float(np.mean([p[0] for p in pts]))})
        geoms.append(geom)
    if not geoms:
        return gpd.GeoDataFrame(columns=["highway", "len_km", "lat", "lon"],
                                geometry=[], crs="EPSG:4326")
    gdf = gpd.GeoDataFrame(recs, geometry=geoms, crs="EPSG:4326")
    metric = gdf.to_crs(gdf.estimate_utm_crs())
    gdf["len_km"] = metric.length.values / 1000.0
    return gdf

def parking_gdf(ways_df):
    """Парковки-way: площадь и центроид."""
    if ways_df.empty:
        return gpd.GeoDataFrame(columns=["area", "lat", "lon"], geometry=[], crs="EPSG:4326")
    mask = ways_df["tags"].apply(lambda t: t.get("amenity") == "parking")
    sub = ways_df[mask]
    if sub.empty:
        return gpd.GeoDataFrame(columns=["area", "lat", "lon"], geometry=[], crs="EPSG:4326")
    recs, geoms = [], []
    for coords in sub["coords"]:
        geom = _safe_polygon(coords)
        if geom is None or geom.is_empty:
            continue
        recs.append({"lat": float(np.mean([p[1] for p in coords])),
                     "lon": float(np.mean([p[0] for p in coords]))})
        geoms.append(geom)
    if not geoms:
        return gpd.GeoDataFrame(columns=["area", "lat", "lon"], geometry=[], crs="EPSG:4326")
    gdf = gpd.GeoDataFrame(recs, geometry=geoms, crs="EPSG:4326")
    metric = gdf.to_crs(gdf.estimate_utm_crs())
    gdf["area"] = metric.area.values
    return gdf

def _to_int(v, default):
    try:
        return max(1, int(float(v)))
    except (TypeError, ValueError):
        return default

# --------------------------------------------------------------------------- #
#  Агрегация в гексы
# --------------------------------------------------------------------------- #
def hex_counts(points_df, res, mask=None):
    """Подсчёт точек по гексам. points_df: lat/lon [+ mask]."""
    if points_df is None or points_df.empty:
        return pd.Series(dtype=float)
    df = points_df if mask is None else points_df[mask]
    if df.empty:
        return pd.Series(dtype=float)
    cells = [h3.latlng_to_cell(a, b, res) for a, b in zip(df["lat"], df["lon"])]
    return pd.Series(cells).value_counts().astype(float)

def mask_by_tag(df, key, values):
    return df["tags"].apply(lambda t: t.get(key) in values)

def populated_with_ring(grid, series):
    """Гексы с value>0 + кольцо соседей (k=1) вокруг них. Нули дальше кольца отбрасываем."""
    populated = set(series[series > 0].index)
    keep = set(populated)
    for cell in populated:
        keep.update(h3.grid_disk(cell, 1))
    return [c for c in grid if c in keep]


def hex_area_km2(res):
    b = h3.cell_to_boundary(h3.latlng_to_cell(55.0, 83.0, res))
    ll = [(lng, lat) for lat, lng in b]
    gdf = gpd.GeoDataFrame(geometry=[Polygon(ll)], crs="EPSG:4326")
    return gdf.to_crs(gdf.estimate_utm_crs()).area.iloc[0] / 1e6

# --------------------------------------------------------------------------- #
#  Контроль карты: ОДНА карта на выбор
# --------------------------------------------------------------------------- #
def compute_series(map_type, sub_option, res, data, kontur_df=None, m2_per_person=30):
    geo, nodes_df, ways_df = data
    wcent = ways_centroids(ways_df)

    if map_type.startswith("1."):
        if kontur_df is not None:  # реальная численность Kontur
            cells = [h3.cell_to_parent(c, res) if h3.get_resolution(c) > res else c
                     for c in kontur_df["h3"]]
            s = kontur_df.assign(cell=cells).groupby("cell")["population"].sum()
            return s, "чел."
        b = buildings_gdf(ways_df)
        if b.empty:
            return pd.Series(dtype=float), "чел. (оценка)"
        cells = [h3.latlng_to_cell(a, b_, res) for a, b_ in zip(b["lat"], b["lon"])]
        vol = (b.assign(cell=cells, vol=b["area"] * b["levels"])
                 .groupby("cell")["vol"].sum())
        return vol / m2_per_person, f"чел. (суррогат, {m2_per_person} м²/чел)"

    if map_type.startswith("2."):
        b = buildings_gdf(ways_df)
        if b.empty:
            return pd.Series(dtype=float), "м²"
        cells = [h3.latlng_to_cell(a, b_, res) for a, b_ in zip(b["lat"], b["lon"])]
        vol = (b.assign(cell=cells, vol=b["area"] * b["levels"])
                 .groupby("cell")["vol"].sum())
        return vol, "м² застройки"

    if map_type.startswith("3."):
        spec = POI_CATEGORIES[sub_option]
        if spec == "__OFFICE_CRAFT__":
            mn = nodes_df["tags"].apply(lambda t: "office" in t or "craft" in t)
            mw = wcent["tags"].apply(lambda t: "office" in t or "craft" in t)
        elif spec is None:  # все POI
            keys = ("amenity", "shop", "craft", "office")
            mn = nodes_df["tags"].apply(lambda t: any(k in t for k in keys))
            mw = wcent["tags"].apply(lambda t: any(k in t for k in keys))
        else:
            key, values = spec
            mn = mask_by_tag(nodes_df, key, values)
            mw = mask_by_tag(wcent, key, values)
        s = pd.concat([hex_counts(nodes_df, res, mn), hex_counts(wcent, res, mw)])
        return s.groupby(level=0).sum(), "объектов"

    if map_type.startswith("4."):
        if sub_option.startswith("Остановки"):
            keys = ("bus_stop", "bus_station", "tram_stop")
            mn = nodes_df["tags"].apply(lambda t: t.get("highway") in keys or
                                                  t.get("public_transport") in ("platform", "stop_position"))
            mw = wcent["tags"].apply(lambda t: t.get("highway") in keys or
                                                   t.get("public_transport") in ("platform", "stop_position"))
            s = pd.concat([hex_counts(nodes_df, res, mn), hex_counts(wcent, res, mw)])
            return s.groupby(level=0).sum(), "остановок"
        if sub_option.startswith("Парковки (кол-во"):
            mn = mask_by_tag(nodes_df, "amenity", {"parking"})
            mw = mask_by_tag(wcent, "amenity", {"parking"})
            s = pd.concat([hex_counts(nodes_df, res, mn), hex_counts(wcent, res, mw)])
            return s.groupby(level=0).sum(), "парковок"
        if sub_option.startswith("Парковки (площадь"):
            p = parking_gdf(ways_df)
            if p.empty:
                return pd.Series(dtype=float), "м²"
            cells = [h3.latlng_to_cell(a, b, res) for a, b in zip(p["lat"], p["lon"])]
            return p.assign(cell=cells).groupby("cell")["area"].sum(), "м² парковок"
        # плотность дорожной сети
        r = roads_gdf(ways_df)
        if r.empty:
            return pd.Series(dtype=float), "км/км²"
        cells = [h3.latlng_to_cell(a, b, res) for a, b in zip(r["lat"], r["lon"])]
        s = r.assign(cell=cells).groupby("cell")["len_km"].sum()
        return s / hex_area_km2(res), "км/км²"

    if map_type.startswith("5."):
        parts = []
        for cat in sub_option:
            key, values = SOCIAL_CATEGORIES[cat]
            parts.append(hex_counts(nodes_df, res, mask_by_tag(nodes_df, key, values)))
            parts.append(hex_counts(wcent, res, mask_by_tag(wcent, key, values)))
        s = pd.concat([p for p in parts if len(p)]).groupby(level=0).sum()
        return s, "объектов"

    if map_type.startswith("6."):
        r = roads_gdf(ways_df)
        if r.empty:
            return pd.Series(dtype=float), "км/км²"
        hw = AUTO_HIGHWAYS if sub_option.startswith("Автомобильный") else PED_HIGHWAYS
        r = r[r["highway"].isin(hw)]
        if r.empty:
            return pd.Series(dtype=float), "км/км²"
        cells = [h3.latlng_to_cell(a, b, res) for a, b in zip(r["lat"], r["lon"])]
        s = r.assign(cell=cells).groupby("cell")["len_km"].sum()
        return s / hex_area_km2(res), "км/км²"

    if map_type.startswith("7."):
        key, values = COMPETITOR_TAGS
        s = pd.concat([hex_counts(nodes_df, res, mask_by_tag(nodes_df, key, values)),
                       hex_counts(wcent, res, mask_by_tag(wcent, key, values))])
        return s.groupby(level=0).sum(), "объектов"

    return pd.Series(dtype=float), ""

# --------------------------------------------------------------------------- #
#  Отрисовка ОДНОЙ карты
# --------------------------------------------------------------------------- #
COLORS = ["#2c7fb8", "#41b6c4", "#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026"]

def render_map(grid, series, unit, geo, map_type):
    center = [geo["lat"], geo["lon"]]
    m = folium.Map(location=center, tiles="OpenStreetMap", control_scale=True)

    vals = series.reindex(grid).fillna(0.0)
    vmax = float(vals.max()) if len(vals) else 0.0
    if vmax <= 0:
        st.warning("Нет данных для выбранного слоя в этом городе.")
        vmax = 1.0
    cm = LinearColormap(COLORS, vmin=0, vmax=vmax)
    cm.caption = f"{map_type} — {unit}"

    # вся сетка — один GeoJSON FeatureCollection (быстро на тысячах гексов)
    feats = []
    for cell in grid:
        boundary = h3.cell_to_boundary(cell)          # [(lat,lng),...]
        ring = [[lng, lat] for lat, lng in boundary]
        ring.append(ring[0])
        feats.append({
            "type": "Feature",
            "properties": {"v": round(float(vals.get(cell, 0.0)), 2)},
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        })
    gj = {"type": "FeatureCollection", "features": feats}

    folium.GeoJson(
        gj,
        style_function=lambda f: {
            "fillColor": cm(f["properties"]["v"]),
            "color": "#555555", "weight": 0.6,
            "fillOpacity": 0.55,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["v"], aliases=[f"{unit}: "], localize=True,
        ),
    ).add_to(m)

    cm.add_to(m)
    bounds = [h3.cell_to_boundary(c) for c in grid]
    lats = [p[0] for b_ in bounds for p in b_]
    lngs = [p[1] for b_ in bounds for p in b_]
    m.fit_bounds([[min(lats), min(lngs)], [max(lats), max(lngs)]])
    st_folium(m, width=1150, height=680, returned_objects=[])


# --------------------------------------------------------------------------- #
#  Kontur Population (локальный файл)
# --------------------------------------------------------------------------- #
def load_kontur(file, res):
    """Файл GeoJSON/GeoParquet Kontur Population -> DataFrame[h3, population]."""
    name = file.name.lower()
    if name.endswith(".parquet"):
        import geopandas as _gpd
        gdf = _gpd.read_parquet(file)
    else:
        gdf = gpd.read_file(file)
    pop_col = next((c for c in gdf.columns if c.lower() in
                    ("population", "pop", "count")), None)
    if pop_col is None:
        st.error("В файле нет колонки population.")
        return None
    gdf = gdf[[pop_col, "geometry"]].rename(columns={pop_col: "population"})
    gdf = gdf[gdf["population"] > 0]
    if "h3" in gdf.columns:
        cells = gdf["h3"].astype(str)
    else:
        cent = gdf.geometry.centroid
        cells = [h3.latlng_to_cell(y, x, res) for y, x in zip(cent.y, cent.x)]
    src_res = h3.get_resolution(cells.iloc[0])
    if res > src_res:
        st.warning(f"Kontur идёт в res{src_res}: показ возможен только при res ≤ {src_res}. "
                   f"Понижаю детализацию до res{src_res}.")
    return pd.DataFrame({"h3": cells, "population": gdf["population"].values})

# --------------------------------------------------------------------------- #
#  UI
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="GeoHex Analytics", layout="wide")
st.title("🗺️ GeoHex Analytics — аналитика города по гексагональной сетке H3")
st.caption("Одна карта на экран. Источники: OpenStreetMap (Overpass API), Nominatim, "
           "опционально Kontur Population. Все API бесплатные.")

# ------------------------------- сайдбар ---------------------------------- #
with st.sidebar:
    st.header("Город")
    city = st.text_input("Введите город", value="Новосибирск")
    load_btn = st.button("🔍 Построить сетку", type="primary")

    st.header("Сетка H3")
    res = st.select_slider(
        "Размер гекса (resolution)",
        options=[7, 8, 9, 10], value=8,
        help="Res 7 ≈ 2,4 км между центрами · Res 8 ≈ 0,92 км · Res 9 ≈ 0,35 км · Res 10 ≈ 0,13 км",
    )

    st.header("Тип карты (одна на экран)")
    map_type = st.radio("Что показываем", MAP_TYPES, index=0)

    sub_option = None
    if map_type.startswith("3."):
        sub_option = st.selectbox("Категория POI", list(POI_CATEGORIES.keys()))
    elif map_type.startswith("4."):
        sub_option = st.selectbox("Показатель", TRANSPORT_MODES)
    elif map_type.startswith("5."):
        sub_option = st.multiselect("Объекты соц. инфраструктуры",
                                    list(SOCIAL_CATEGORIES.keys()),
                                    default=list(SOCIAL_CATEGORIES.keys()))
    elif map_type.startswith("6."):
        sub_option = st.selectbox("Вид трафика", TRAFFIC_MODES)

    m2_per_person = 30
    kontur_df = None
    if map_type.startswith("1."):
        src = st.radio("Источник численности",
                       ["Суррогатная оценка (OSM-здания)", "Kontur Population (файл)"])
        if src.startswith("Суррогат"):
            m2_per_person = st.slider("Норма м² жилья на человека", 15, 60, 30)
        else:
            f = st.file_uploader("Файл Kontur Population (.geojson / .parquet)",
                                 type=["geojson", "json", "parquet"])
            if f is not None:
                with st.spinner("Загружаю Kontur Population…"):
                    kontur_df = load_kontur(f, res)

# ------------------------------- логика ----------------------------------- #
if load_btn:
    with st.spinner(f"Геокодирую «{city}»…"):
        geo = geocode_city(city)
    if geo is None:
        st.error("Город не найден. Уточните название.")
        st.stop()
    with st.spinner("Загружаю данные OpenStreetMap через Overpass API (это может занять 1–5 минут)…"):
        try:
            geo, nodes_df, ways_df = fetch_city_data(city)
        except RuntimeError as e:
            st.error(str(e))
            st.stop()
    st.session_state["data"] = (geo, nodes_df, ways_df)

if "data" not in st.session_state:
    st.info("Введите город в боковой панели и нажмите «Построить сетку».")
    st.stop()

data = st.session_state["data"]
geo, nodes_df, ways_df = data
st.success(f"📍 {geo['display']}")

grid, in_boundary = make_grid(geo, res)
if not in_boundary:
    st.info("Граница города не получена от Nominatim (лимит 0,5 МБ) — сетка построена "
            "по прямоугольной области. Уточните название города или повторите попытку.")
if len(grid) > MAX_GRID_CELLS:
    st.error(f"Сетка слишком велика ({len(grid)} гексов при res {res}, "
             f"лимит {MAX_GRID_CELLS}). Понизьте resolution.")
    st.stop()

with st.spinner("Считаю агрегаты по гексам…"):
    series, unit = compute_series(map_type, sub_option, res, data,
                                  kontur_df=kontur_df, m2_per_person=m2_per_person)

# суррогатные карты: не рисуем гексы с 0, кроме кольца вокруг заселённых
if map_type.startswith(("1.", "2.")):
    grid = populated_with_ring(grid, series)

c1, c2, c3 = st.columns(3)
c1.metric("Гексов в сетке", f"{len(grid):,}")
c2.metric("Resolution", f"res {res} (~{RES_SPACING_KM[res]} км между центрами)")
c3.metric("Максимум в ячейке", f"{series.max():,.0f} {unit}" if len(series) else "—")

render_map(grid, series, unit, geo, map_type)
st.caption("⚠️ Оценки по OSM-зданиям — суррогатные: не учитывают реальное заселение и "
           "незавершённое строительство. Для точной численности загрузите Kontur Population "
           "(data.humdata.org, датасет «Kontur Population»).")
