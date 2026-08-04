# -*- coding: utf-8 -*-
"""
🏥 GeoClinic Analyst — Streamlit-версия
Анализ локации под многофункциональную клинику
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import geopandas as gpd
import h3
import osmnx as ox
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from shapely.geometry import Polygon
import branca.colormap as cm
import numpy as np
import time

# ═══════════════════════════════════════════════════════════════
# НАСТРОЙКА СТРАНИЦЫ
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="GeoClinic Analyst",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }
    .stProgress > div > div > div > div {
        background-color: #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# КОНСТАНТЫ
# ═══════════════════════════════════════════════════════════════
H3_RESOLUTION = 9  # ~182 м
AMENITY_TRANSLATION = {
    "hospital": "Больницы",
    "clinic": "Клиники и медцентры",
    "doctors": "Врачебные кабинеты",
    "dentist": "Стоматологии",
    "pharmacy": "Аптеки",
}

# ═══════════════════════════════════════════════════════════════
# КЭШИРУЕМЫЕ ФУНКЦИИ (OSM-запросы)
# ═══════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def geocode_address(address: str):
    """Геокодирование адреса через Nominatim."""
    geolocator = Nominatim(user_agent="geoclinic_analyst_streamlit_v1")
    try:
        location = geolocator.geocode(address, timeout=10)
        if location:
            return {
                "lat": location.latitude,
                "lng": location.longitude,
                "address": location.address,
            }
    except GeocoderTimedOut:
        time.sleep(1)
        location = geolocator.geocode(address, timeout=15)
        if location:
            return {
                "lat": location.latitude,
                "lng": location.longitude,
                "address": location.address,
            }
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_osm_features(center_lat, center_lng, radius_m, tags):
    """Универсальная загрузка OSM-данных с fallback."""
    try:
        gdf = ox.features_from_point((center_lat, center_lng), tags=tags, dist=radius_m)
        return gdf
    except Exception:
        try:
            gdf = ox.geometries_from_point((center_lat, center_lng), tags=tags, dist=radius_m)
            return gdf
        except Exception as e:
            st.warning(f"Ошибка загрузки OSM: {e}")
            return gpd.GeoDataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def get_medical_df(center_lat, center_lng, radius_m):
    """Медицинские объекты."""
    tags = {"amenity": list(AMENITY_TRANSLATION.keys())}
    gdf = fetch_osm_features(center_lat, center_lng, radius_m, tags)
    if gdf.empty:
        return pd.DataFrame(columns=["lat", "lng", "type_en", "type_ru", "name"])
    rows = []
    for idx, row in gdf.iterrows():
        centroid = row.geometry.centroid
        name = row.get("name", "Без названия")
        if pd.isna(name):
            name = "Без названия"
        rows.append({
            "lat": centroid.y,
            "lng": centroid.x,
            "type_en": row.amenity,
            "type_ru": AMENITY_TRANSLATION.get(row.amenity, row.amenity),
            "name": name,
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600, show_spinner=False)
def get_buildings_df(center_lat, center_lng, radius_m):
    """Жилые здания с оценкой населения."""
    tags = {"building": ["apartments", "residential", "house", "living_quarter"]}
    gdf = fetch_osm_features(center_lat, center_lng, radius_m, tags)
    if gdf.empty:
        return pd.DataFrame(columns=["hex_id", "people", "levels", "building_type", "area_m2"])
    gdf_meters = gdf.to_crs(epsg=3857)
    rows = []
    for idx, row in gdf.iterrows():
        centroid = row.geometry.centroid
        hex_id = h3.latlng_to_cell(centroid.y, centroid.x, H3_RESOLUTION)
        try:
            footprint_area = gdf_meters.loc[idx].geometry.area
        except Exception:
            footprint_area = 0
        levels = row.get("building:levels", None)
        b_type = row.get("building", "residential")
        if pd.isna(levels) or not str(levels).isdigit():
            levels = 9 if b_type == "apartments" else 5
        else:
            levels = int(levels)
        total_area = footprint_area * levels
        estimated = max(2, int(total_area / 27)) if total_area > 0 else 2
        rows.append({
            "hex_id": hex_id,
            "people": estimated,
            "levels": levels,
            "building_type": b_type,
            "area_m2": total_area,
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600, show_spinner=False)
def get_wealth_df(center_lat, center_lng, radius_m):
    """Индекс платёжеспособности."""
    tags = {
        "building": ["apartments", "residential", "house"],
        "amenity": ["bank", "atm", "restaurant", "cafe"],
        "shop": ["mall", "boutique"],
    }
    gdf = fetch_osm_features(center_lat, center_lng, radius_m, tags)
    if gdf.empty:
        return pd.DataFrame(columns=["hex_id", "score"])
    rows = []
    for idx, row in gdf.iterrows():
        centroid = row.geometry.centroid
        hex_id = h3.latlng_to_cell(centroid.y, centroid.x, H3_RESOLUTION)
        score = 0
        b_type = row.get("building", None)
        if pd.notna(b_type):
            levels = row.get("building:levels", 5)
            try:
                levels = int(levels) if str(levels).isdigit() else 5
            except:
                levels = 5
            if b_type == "apartments":
                score += levels * 3
            elif b_type == "residential":
                score += levels * 1.5
            elif b_type == "house":
                score += 1
        amenity = row.get("amenity", None)
        if pd.notna(amenity) and amenity in ["bank", "restaurant"]:
            score += 15
        shop = row.get("shop", None)
        if pd.notna(shop) and shop in ["mall", "boutique"]:
            score += 25
        if score > 0:
            rows.append({"hex_id": hex_id, "score": score})
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600, show_spinner=False)
def get_traffic_df(center_lat, center_lng, radius_m):
    """Трафик: авто + пешеходный."""
    tags = {
        "highway": ["trunk", "primary", "secondary", "tertiary", "residential",
                    "footway", "pedestrian", "crossing", "traffic_signals"],
        "amenity": ["bus_station", "fuel", "parking"],
        "public_transport": ["platform", "stop_position"],
        "shop": ["supermarket", "convenience", "mall"],
    }
    gdf = fetch_osm_features(center_lat, center_lng, radius_m, tags)
    auto_rows, ped_rows = [], []
    if gdf.empty:
        return pd.DataFrame(columns=["hex_id", "auto_score"]), pd.DataFrame(columns=["hex_id", "ped_score"])
    for idx, row in gdf.iterrows():
        centroid = row.geometry.centroid
        hex_id = h3.latlng_to_cell(centroid.y, centroid.x, H3_RESOLUTION)
        highway = row.get("highway", None)
        amenity = row.get("amenity", None)
        shop = row.get("shop", None)
        pt = row.get("public_transport", None)
        # Авто
        if pd.notna(highway):
            if highway in ["trunk", "primary"]:
                auto_rows.append({"hex_id": hex_id, "score": 50})
            elif highway in ["secondary", "tertiary"]:
                auto_rows.append({"hex_id": hex_id, "score": 25})
            elif highway == "traffic_signals":
                auto_rows.append({"hex_id": hex_id, "score": 15})
        if pd.notna(amenity) and amenity in ["fuel", "parking"]:
            auto_rows.append({"hex_id": hex_id, "score": 20})
        # Пешеходный
        if pd.notna(highway) and highway in ["footway", "pedestrian", "crossing"]:
            ped_rows.append({"hex_id": hex_id, "score": 30})
        if pd.notna(pt) or (pd.notna(amenity) and amenity == "bus_station"):
            ped_rows.append({"hex_id": hex_id, "score": 40})
        if pd.notna(shop) and shop in ["supermarket", "convenience", "mall"]:
            ped_rows.append({"hex_id": hex_id, "score": 35})
    df_auto = pd.DataFrame(auto_rows).groupby("hex_id")["score"].sum().reset_index() if auto_rows else pd.DataFrame(columns=["hex_id", "score"])
    df_ped = pd.DataFrame(ped_rows).groupby("hex_id")["score"].sum().reset_index() if ped_rows else pd.DataFrame(columns=["hex_id", "score"])
    df_auto.rename(columns={"score": "auto_score"}, inplace=True)
    df_ped.rename(columns={"score": "ped_score"}, inplace=True)
    return df_auto, df_ped


@st.cache_data(ttl=3600, show_spinner=False)
def get_pvz_df(center_lat, center_lng, radius_m):
    """Пункты выдачи (Ozon, WB, Яндекс, СДЭК)."""
    tags_pvz = {
        "brand": ["Ozon", "Wildberries", "Яндекс Маркет", "Яндекс.Маркет",
                  "озон", "вайлдберриз", "СДЭК", "CDEK", "sdek", "сдэк", "WB", "ВБ", "яндекс"],
        "shop": ["parcel_pickup", "delivery"],
        "amenity": ["parcel_pickup"],
    }
    gdf = fetch_osm_features(center_lat, center_lng, radius_m, tags_pvz)
    if gdf.empty:
        return pd.DataFrame(columns=["lat", "lng", "brand"])
    rows = []
    for idx, row in gdf.iterrows():
        centroid = row.geometry.centroid
        brand_raw = str(row.get("brand", "")).lower()
        name_raw = str(row.get("name", "")).lower()
        det = "Другой ПВЗ"
        if "ozon" in brand_raw or "озон" in brand_raw or "ozon" in name_raw or "озон" in name_raw:
            det = "Ozon"
        elif "wildberries" in brand_raw or "вайлдберриз" in brand_raw or "wb" in brand_raw or "wildberries" in name_raw:
            det = "Wildberries"
        elif "яндекс" in brand_raw or "yandex" in brand_raw or "яндекс" in name_raw:
            det = "Яндекс Маркет"
        elif "сдэк" in brand_raw or "cdek" in brand_raw or "sdek" in brand_raw or "сдэк" in name_raw:
            det = "СДЭК"
        rows.append({"lat": centroid.y, "lng": centroid.x, "brand": det})
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ═══════════════════════════════════════════════════════════════

def create_hex_grid(center_lat, center_lng, radius_m, resolution=H3_RESOLUTION):
    """Создаёт GeoDataFrame с гексагональной сеткой."""
    center_hex = h3.latlng_to_cell(center_lat, center_lng, resolution)
    max_rings = max(1, int(radius_m / 180))
    all_hexes = h3.grid_disk(center_hex, max_rings)
    df_hex = pd.DataFrame({"hex_id": list(all_hexes)})
    def hex_to_geo(hex_id):
        boundary = h3.cell_to_boundary(hex_id)
        return Polygon([(lng, lat) for lat, lng in boundary])
    df_hex["geometry"] = df_hex["hex_id"].apply(hex_to_geo)
    return gpd.GeoDataFrame(df_hex, geometry="geometry", crs="EPSG:4326")


def get_brand_color(brand):
    mapping = {
        "Ozon": "#005bff",
        "Wildberries": "#8a2be2",
        "Яндекс Маркет": "#ff0000",
        "СДЭК": "#27ae60",
    }
    return mapping.get(brand, "#7f8c8d")


def make_combined_tooltip(row):
    lines = [
        "<b>Статистика ячейки:</b>",
        f"Аптеки: {row.get('count_pharmacy', 0)}",
        f"Стоматологии: {row.get('count_dentist', 0)}",
        f"Клиники и медцентры: {row.get('count_clinic', 0)}",
        f"Больницы: {row.get('count_hospital', 0)}",
        f"Врачебные кабинеты: {row.get('count_doctors', 0)}",
    ]
    return "<br>".join(lines)


def get_wealth_label(score, max_s):
    if max_s == 0:
        max_s = 1
    ratio = score / max_s
    if score == 0:
        return "Низкая застройка / Промзона"
    elif ratio < 0.2:
        return "Ниже среднего (Частный сектор / Малоэтажки)"
    elif ratio < 0.5:
        return "Средний класс (Советские панели / Пятиэтажки)"
    elif ratio < 0.8:
        return "Выше среднего (Новостройки / Районы с кафе)"
    else:
        return "Высокая (Элитные ЖК / ТРЦ / Бизнес-центры)"


# ═══════════════════════════════════════════════════════════════
# ФУНКЦИИ ПОСТРОЕНИЯ КАРТ
# ═══════════════════════════════════════════════════════════════

def build_medical_map(center_lat, center_lng, radius_m, df_med, gdf_hex_base):
    m = folium.Map(location=[center_lat, center_lng], zoom_start=13, tiles="OpenStreetMap")
    folium.Marker(
        [center_lat, center_lng],
        popup="Центр анализа",
        icon=folium.Icon(color="red", icon="home"),
    ).add_to(m)
    folium.Circle(
        radius=radius_m, location=[center_lat, center_lng],
        color="crimson", fill=False, weight=2, dash_array="5, 5",
    ).add_to(m)
    for ent_en, ent_ru in AMENITY_TRANSLATION.items():
        df_sub = df_med[df_med["type_en"] == ent_en] if not df_med.empty else pd.DataFrame()
        gdf_hex = gdf_hex_base.copy()
        gdf_hex["current_count"] = gdf_hex[f"count_{ent_en}"]
        max_val = int(gdf_hex["current_count"].max()) if len(gdf_hex) > 0 else 1
        if max_val == 0:
            max_val = 1
        colormap = cm.linear.YlOrRd_09.scale(0, max_val)
        fg = folium.FeatureGroup(name=f"Плотность: {ent_ru}", overlay=True, control=True)
        custom_tooltip = folium.GeoJsonTooltip(
            fields=["tooltip_text"], aliases=[""], sticky=True,
            style="background-color: #F0EFEF; border: 2px solid black; border-radius: 3px; font-size: 12px; padding: 5px;"
        )
        folium.GeoJson(
            gdf_hex.to_json(),
            style_function=lambda feature, clr=colormap: {
                "fillColor": clr(feature["properties"]["current_count"]) if feature["properties"]["current_count"] > 0 else "transparent",
                "color": "black", "weight": 0.01,
                "fillOpacity": 0.4 if feature["properties"]["current_count"] > 0 else 0.0
            },
            tooltip=custom_tooltip,
        ).add_to(fg)
        for _, row in df_sub.iterrows():
            folium.CircleMarker(
                location=[row["lat"], row["lng"]],
                radius=3, color="blue", fill=True, fill_color="blue", fill_opacity=0.8,
                popup=f"<b>{row['name']}</b><br>{row['type_ru']}",
            ).add_to(fg)
        fg.add_to(m)
    folium.LayerControl(collapsed=False, position="topright").add_to(m)
    return m


def build_population_map(center_lat, center_lng, radius_m, gdf_hex):
    m = folium.Map(location=[center_lat, center_lng], zoom_start=12, tiles="OpenStreetMap")
    folium.Marker(
        [center_lat, center_lng],
        popup="Центр анализа",
        icon=folium.Icon(color="red", icon="home"),
    ).add_to(m)
    folium.Circle(
        radius=radius_m, location=[center_lat, center_lng],
        color="crimson", fill=False, weight=2, dash_array="5, 5",
    ).add_to(m)
    min_val = int(gdf_hex["pop_count"].min())
    max_val = int(gdf_hex["pop_count"].max())
    if min_val == max_val:
        max_val = min_val + 1
    colormap = cm.linear.Purples_09.scale(min_val, max_val)
    folium.GeoJson(
        gdf_hex.to_json(),
        style_function=lambda feature, clr=colormap: {
            "fillColor": clr(feature["properties"]["pop_count"]) if feature["properties"]["pop_count"] > 0 else "transparent",
            "color": "gray", "weight": 0.2,
            "fillOpacity": 0.6 if feature["properties"]["pop_count"] > 0 else 0.0
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["tooltip_text"], aliases=[""], sticky=True,
            style="background-color: #F0EFEF; border: 1px solid black; font-size: 12px; font-weight: bold; padding: 5px;"
        ),
    ).add_to(m)
    colormap.caption = "Плотность населения (чел. в гексагоне ~0.1 кв.км)"
    colormap.add_to(m)
    return m


def build_wealth_map(center_lat, center_lng, radius_m, gdf_hex):
    m = folium.Map(location=[center_lat, center_lng], zoom_start=12, tiles="OpenStreetMap")
    folium.Marker(
        [center_lat, center_lng],
        popup="Центр анализа",
        icon=folium.Icon(color="red", icon="home"),
    ).add_to(m)
    folium.Circle(
        radius=radius_m, location=[center_lat, center_lng],
        color="crimson", fill=False, weight=2, dash_array="5, 5",
    ).add_to(m)
    min_val = int(gdf_hex["wealth_score"].min())
    max_val = int(gdf_hex["wealth_score"].max())
    if min_val == max_val:
        max_val = min_val + 1
    colormap = cm.linear.YlGn_09.scale(min_val, max_val)
    folium.GeoJson(
        gdf_hex.to_json(),
        style_function=lambda feature, clr=colormap: {
            "fillColor": clr(feature["properties"]["wealth_score"]) if feature["properties"]["wealth_score"] > 0 else "transparent",
            "color": "gray", "weight": 0.2,
            "fillOpacity": 0.6 if feature["properties"]["wealth_score"] > 0 else 0.0
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["tooltip_text"], aliases=[""], sticky=True,
            style="background-color: #F0EFEF; border: 1px solid black; font-size: 12px; font-weight: bold; padding: 5px;"
        ),
    ).add_to(m)
    colormap.caption = "Относительный уровень платёжеспособности"
    colormap.add_to(m)
    return m


def build_traffic_map(center_lat, center_lng, radius_m, gdf_hex_base):
    m = folium.Map(location=[center_lat, center_lng], zoom_start=13, tiles="OpenStreetMap")
    folium.Marker(
        [center_lat, center_lng],
        popup="Центр анализа",
        icon=folium.Icon(color="red", icon="home"),
    ).add_to(m)
    folium.Circle(
        radius=radius_m, location=[center_lat, center_lng],
        color="crimson", fill=False, weight=2, dash_array="5, 5",
    ).add_to(m)
    max_auto = gdf_hex_base["auto_score"].max() if gdf_hex_base["auto_score"].max() > 0 else 1
    max_ped = gdf_hex_base["ped_score"].max() if gdf_hex_base["ped_score"].max() > 0 else 1
    colormap_auto = cm.linear.Oranges_09.scale(0, max_auto)
    colormap_ped = cm.linear.Blues_09.scale(0, max_ped)
    fg_auto = folium.FeatureGroup(name="Автомобильный трафик (Оранжевый)", overlay=False, control=True)
    folium.GeoJson(
        gdf_hex_base.to_json(),
        style_function=lambda feature: {
            "fillColor": colormap_auto(feature["properties"]["auto_score"]) if feature["properties"]["auto_score"] > 0 else "transparent",
            "color": "black", "weight": 0.01,
            "fillOpacity": 0.5 if feature["properties"]["auto_score"] > 0 else 0.0
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["tooltip_text"], aliases=[""], sticky=True,
            style="background-color: #F0EFEF; border: 2px solid black; border-radius: 3px; font-size: 12px; font-weight: bold; padding: 5px;"
        ),
    ).add_to(fg_auto)
    fg_auto.add_to(m)
    fg_ped = folium.FeatureGroup(name="Пешеходный трафик (Синий)", overlay=False, control=True)
    folium.GeoJson(
        gdf_hex_base.to_json(),
        style_function=lambda feature: {
            "fillColor": colormap_ped(feature["properties"]["ped_score"]) if feature["properties"]["ped_score"] > 0 else "transparent",
            "color": "black", "weight": 0.01,
            "fillOpacity": 0.5 if feature["properties"]["ped_score"] > 0 else 0.0
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["tooltip_text"], aliases=[""], sticky=True,
            style="background-color: #F0EFEF; border: 2px solid black; border-radius: 3px; font-size: 12px; font-weight: bold; padding: 5px;"
        ),
    ).add_to(fg_ped)
    fg_ped.add_to(m)
    folium.LayerControl(collapsed=False, position="topright").add_to(m)
    return m


def build_pvz_map(center_lat, center_lng, radius_m, df_pvz, gdf_hex_base):
    m = folium.Map(location=[center_lat, center_lng], zoom_start=13, tiles="OpenStreetMap")
    folium.Marker(
        [center_lat, center_lng],
        popup="Центр анализа",
        icon=folium.Icon(color="red", icon="home"),
    ).add_to(m)
    folium.Circle(
        radius=radius_m, location=[center_lat, center_lng],
        color="crimson", fill=False, weight=2, dash_array="5, 5",
    ).add_to(m)
    max_val = int(gdf_hex_base["pvz_count"].max())
    if max_val == 0:
        max_val = 1
    colormap = cm.linear.Greys_09.scale(0, max_val)
    folium.GeoJson(
        gdf_hex_base.to_json(),
        style_function=lambda feature, clr=colormap: {
            "fillColor": clr(feature["properties"]["pvz_count"]) if feature["properties"]["pvz_count"] > 0 else "transparent",
            "color": "black", "weight": 0.15,
            "fillOpacity": 0.35 if feature["properties"]["pvz_count"] > 0 else 0.0
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["tooltip_text"], aliases=[""], sticky=True,
            style="background-color: #F0EFEF; border: 2px solid black; font-size: 12px; font-weight: bold; padding: 5px;"
        ),
    ).add_to(m)
    for _, row in df_pvz.iterrows():
        color = get_brand_color(row["brand"])
        folium.CircleMarker(
            location=[row["lat"], row["lng"]],
            radius=4.5, color="black", weight=0.5,
            fill=True, fill_color=color, fill_opacity=0.9,
            popup=f"<b>{row['brand']}</b><br>Пункт выдачи/доставки",
        ).add_to(m)
    return m


# ═══════════════════════════════════════════════════════════════
# БОКОВАЯ ПАНЕЛЬ
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2964/2964514.png", width=80)
    st.markdown("## 🏥 GeoClinic Analyst")
    st.markdown("Анализ локации под многофункциональную клинику")
    st.divider()

    address = st.text_input(
        "📍 Адрес объекта",
        value="Город, улица, дом",
        help="Введите полный адрес для геокодирования. Например: Москва, Арбат, 10",
    )
    radius_km = st.slider(
        "🔍 Радиус анализа, км",
        min_value=1, max_value=5, value=3, step=1,
        help="Максимальный радиус анализа вокруг точки",
    )

    st.divider()
    st.markdown("### 🎯 Целевая аудитория")

    target_age = st.number_input(
        "Средний возраст, лет",
        min_value=18, max_value=90, value=40, step=1,
        help="Средний возраст целевой аудитории клиники",
    )
    women_pct = st.slider(
        "Доля женщин, %",
        min_value=0, max_value=100, value=60, step=5,
        help="Остаток автоматически идёт на мужчин",
    )
    men_pct = 100 - women_pct
    st.markdown(f"- Мужчины: **{men_pct}%**")

    st.divider()
    run_analysis = st.button("🚀 Запустить анализ", use_container_width=True, type="primary")

# ═══════════════════════════════════════════════════════════════
# ГЛАВНАЯ ОБЛАСТЬ
# ═══════════════════════════════════════════════════════════════
st.markdown('<div class="main-header">🏥 GeoClinic Analyst</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Геомаркетинговая аналитика для открытия многофункциональной клиники</div>', unsafe_allow_html=True)

if not run_analysis:
    st.info("👈 Введите адрес и параметры ЦА в боковой панели, затем нажмите **Запустить анализ**.")
    st.stop()

# ═══════════════════════════════════════════════════════════════
# ШАГ 1: ГЕОКОДИРОВАНИЕ
# ═══════════════════════════════════════════════════════════════
progress_bar = st.progress(0, text="📍 Геокодирование адреса...")
geo_result = geocode_address(address)
if geo_result is None:
    st.error("❌ Не удалось найти адрес. Проверьте правильность написания.")
    st.stop()

CENTER_LAT = geo_result["lat"]
CENTER_LNG = geo_result["lng"]
RADIUS_METER = radius_km * 1000

progress_bar.progress(10, text=f"✅ Адрес найден: {geo_result['address']}")

# Метрики сверху
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Широта", f"{CENTER_LAT:.5f}")
with col2:
    st.metric("Долгота", f"{CENTER_LNG:.5f}")
with col3:
    st.metric("Радиус", f"{radius_km} км")
with col4:
    st.metric("ЦА", f"{target_age} лет, {women_pct}%Ж")

st.divider()

# ═══════════════════════════════════════════════════════════════
# БАЗОВАЯ СЕТКА ГЕКСАГОНОВ
# ═══════════════════════════════════════════════════════════════
gdf_hex_base = create_hex_grid(CENTER_LAT, CENTER_LNG, RADIUS_METER, H3_RESOLUTION)

# ═══════════════════════════════════════════════════════════════
# МОДУЛЬ 1: МЕДИЦИНСКИЕ УЧРЕЖДЕНИЯ
# ═══════════════════════════════════════════════════════════════
progress_bar.progress(20, text="🏥 Загрузка медицинских объектов...")
df_med = get_medical_df(CENTER_LAT, CENTER_LNG, RADIUS_METER)

# Агрегация по гексагонам
for ent_en in AMENITY_TRANSLATION.keys():
    df_sub = df_med[df_med["type_en"] == ent_en] if not df_med.empty else pd.DataFrame()
    if not df_sub.empty:
        df_sub = df_sub.copy()
        df_sub["point_hex"] = df_sub.apply(lambda r: h3.latlng_to_cell(r["lat"], r["lng"], H3_RESOLUTION), axis=1)
        counts = df_sub["point_hex"].value_counts()
        gdf_hex_base[f"count_{ent_en}"] = gdf_hex_base["hex_id"].map(counts).fillna(0).astype(int)
    else:
        gdf_hex_base[f"count_{ent_en}"] = 0

gdf_hex_base["tooltip_text"] = gdf_hex_base.apply(make_combined_tooltip, axis=1)

st.subheader("🩺 Медицинская инфраструктура и конкуренция")
st.caption(f"Найдено медицинских объектов: **{len(df_med)}** в радиусе {radius_km} км")
med_cols = st.columns([3, 1])
with med_cols[0]:
    med_map = build_medical_map(CENTER_LAT, CENTER_LNG, RADIUS_METER, df_med, gdf_hex_base)
    st_folium(med_map, width=700, height=500, returned_objects=[])
with med_cols[1]:
    st.markdown("**Распределение по типам:**")
    if not df_med.empty:
        type_counts = df_med["type_ru"].value_counts()
        for t, c in type_counts.items():
            st.markdown(f"- {t}: **{c}**")
    else:
        st.info("Медицинских объектов не найдено.")

    total_med = len(df_med)
    hex_with_med = (gdf_hex_base[[f"count_{k}" for k in AMENITY_TRANSLATION.keys()]].sum(axis=1) > 0).sum()
    st.metric("Занятых ячеек", f"{hex_with_med}")
    if total_med > 15:
        st.error("⚠️ Высокая конкуренция: >15 объектов")
    elif total_med > 8:
        st.warning("⚠️ Средняя конкуренция: 8–15 объектов")
    else:
        st.success("✅ Низкая конкуренция: <8 объектов")

st.divider()

# ═══════════════════════════════════════════════════════════════
# МОДУЛЬ 2: ПЛОТНОСТЬ НАСЕЛЕНИЯ
# ═══════════════════════════════════════════════════════════════
progress_bar.progress(40, text="👥 Оценка плотности населения...")
df_pop = get_buildings_df(CENTER_LAT, CENTER_LNG, RADIUS_METER)

gdf_hex_pop = create_hex_grid(CENTER_LAT, CENTER_LNG, RADIUS_METER, H3_RESOLUTION)
if not df_pop.empty:
    pop_counts = df_pop.groupby("hex_id")["people"].sum()
    gdf_hex_pop["pop_count"] = gdf_hex_pop["hex_id"].map(pop_counts).fillna(0).astype(int)
else:
    gdf_hex_pop["pop_count"] = 0

gdf_hex_pop["tooltip_text"] = gdf_hex_pop["pop_count"].apply(lambda x: f"Примерное население ячейки: {x} чел.")

total_pop = int(gdf_hex_pop["pop_count"].sum())
max_pop_hex = int(gdf_hex_pop["pop_count"].max())

st.subheader("👥 Плотность населения")
st.caption(f"Оценочное население в зоне анализа: **~{total_pop:,} чел.** | Пик в ячейке: **{max_pop_hex} чел.**")
pop_cols = st.columns([3, 1])
with pop_cols[0]:
    pop_map = build_population_map(CENTER_LAT, CENTER_LNG, RADIUS_METER, gdf_hex_pop)
    st_folium(pop_map, width=700, height=500, returned_objects=[])
with pop_cols[1]:
    st.metric("Всего жителей", f"~{total_pop:,}")
    st.metric("Жилых строений", len(df_pop))
    st.metric("Пик плотности", f"{max_pop_hex} чел./яч.")
    if total_pop > 50000:
        st.success("✅ Отличная плотность для клиники")
    elif total_pop > 20000:
        st.info("ℹ️ Достаточная плотность")
    else:
        st.warning("⚠️ Низкая плотность — риск недозагрузки")

st.divider()

# ═══════════════════════════════════════════════════════════════
# МОДУЛЬ 3: ИНДЕКС ПЛАТЁЖЕСПОСОБНОСТИ
# ═══════════════════════════════════════════════════════════════
progress_bar.progress(60, text="💰 Анализ платёжеспособности...")
df_wealth = get_wealth_df(CENTER_LAT, CENTER_LNG, RADIUS_METER)

gdf_hex_wealth = create_hex_grid(CENTER_LAT, CENTER_LNG, RADIUS_METER, H3_RESOLUTION)
if not df_wealth.empty:
    total_scores = df_wealth.groupby("hex_id")["score"].sum()
    gdf_hex_wealth["wealth_score"] = gdf_hex_wealth["hex_id"].map(total_scores).fillna(0).astype(int)
else:
    gdf_hex_wealth["wealth_score"] = 0

max_score = gdf_hex_wealth["wealth_score"].max() if gdf_hex_wealth["wealth_score"].max() > 0 else 1
gdf_hex_wealth["status"] = gdf_hex_wealth["wealth_score"].apply(lambda x: get_wealth_label(x, max_score))
gdf_hex_wealth["tooltip_text"] = gdf_hex_wealth.apply(lambda r: f"Индекс спроса: {r['wealth_score']} ({r['status']})", axis=1)

avg_wealth = int(gdf_hex_wealth["wealth_score"].mean())
high_wealth_cells = int((gdf_hex_wealth["wealth_score"] > max_score * 0.5).sum())

st.subheader("💰 Индекс спроса (платёжеспособность)")
st.caption("На основе класса недвижимости, этажности и коммерческой инфраструктуры")
wealth_cols = st.columns([3, 1])
with wealth_cols[0]:
    wealth_map = build_wealth_map(CENTER_LAT, CENTER_LNG, RADIUS_METER, gdf_hex_wealth)
    st_folium(wealth_map, width=700, height=500, returned_objects=[])
with wealth_cols[1]:
    st.metric("Средний индекс", avg_wealth)
    st.metric("Премиальных ячеек", high_wealth_cells)
    status_dist = gdf_hex_wealth["status"].value_counts()
    st.markdown("**Распределение:**")
    for s, c in status_dist.head(4).items():
        st.markdown(f"- {s}: **{c}**")
    if high_wealth_cells > 5:
        st.success("✅ Хороший платёжеспособный спрос")
    else:
        st.warning("⚠️ Мало премиальных ячеек")

st.divider()

# ═══════════════════════════════════════════════════════════════
# МОДУЛЬ 4: ТРАФИК
# ═══════════════════════════════════════════════════════════════
progress_bar.progress(80, text="🚗 Анализ трафика...")
df_auto, df_ped = get_traffic_df(CENTER_LAT, CENTER_LNG, RADIUS_METER)

gdf_hex_traffic = create_hex_grid(CENTER_LAT, CENTER_LNG, RADIUS_METER, H3_RESOLUTION)
if not df_auto.empty:
    auto_map_s = df_auto.set_index("hex_id")["auto_score"]
    gdf_hex_traffic["auto_score"] = gdf_hex_traffic["hex_id"].map(auto_map_s).fillna(0).astype(int)
else:
    gdf_hex_traffic["auto_score"] = 0
if not df_ped.empty:
    ped_map_s = df_ped.set_index("hex_id")["ped_score"]
    gdf_hex_traffic["ped_score"] = gdf_hex_traffic["hex_id"].map(ped_map_s).fillna(0).astype(int)
else:
    gdf_hex_traffic["ped_score"] = 0

gdf_hex_traffic["tooltip_text"] = gdf_hex_traffic.apply(
    lambda r: f"Пешеходный трафик: {r['ped_score']}<br>Автомобильный трафик: {r['auto_score']}", axis=1
)

max_auto = gdf_hex_traffic["auto_score"].max() if gdf_hex_traffic["auto_score"].max() > 0 else 1
max_ped = gdf_hex_traffic["ped_score"].max() if gdf_hex_traffic["ped_score"].max() > 0 else 1

st.subheader("🚗 Трафик локации")
st.caption("Автомобильный и пешеходный трафик на основе дорожной инфраструктуры")
traf_cols = st.columns([3, 1])
with traf_cols[0]:
    traffic_map = build_traffic_map(CENTER_LAT, CENTER_LNG, RADIUS_METER, gdf_hex_traffic)
    st_folium(traffic_map, width=700, height=500, returned_objects=[])
with traf_cols[1]:
    st.metric("Пеш. трафик (макс)", int(max_ped))
    st.metric("Авто трафик (макс)", int(max_auto))
    if max_ped > 100 and max_auto > 50:
        st.success("✅ Отличная транспортная доступность")
    elif max_ped > 50 or max_auto > 30:
        st.info("ℹ️ Достаточная доступность")
    else:
        st.warning("⚠️ Слабая транспортная доступность")

st.divider()

# ═══════════════════════════════════════════════════════════════
# МОДУЛЬ 5: ПВЗ (ЛОГИСТИКА)
# ═══════════════════════════════════════════════════════════════
progress_bar.progress(95, text="📦 Поиск пунктов выдачи...")
df_pvz = get_pvz_df(CENTER_LAT, CENTER_LNG, RADIUS_METER)

gdf_hex_pvz = create_hex_grid(CENTER_LAT, CENTER_LNG, RADIUS_METER, H3_RESOLUTION)
if not df_pvz.empty:
    df_pvz_copy = df_pvz.copy()
    df_pvz_copy["point_hex"] = df_pvz_copy.apply(lambda r: h3.latlng_to_cell(r["lat"], r["lng"], H3_RESOLUTION), axis=1)
    counts = df_pvz_copy["point_hex"].value_counts()
    gdf_hex_pvz["pvz_count"] = gdf_hex_pvz["hex_id"].map(counts).fillna(0).astype(int)
else:
    gdf_hex_pvz["pvz_count"] = 0

gdf_hex_pvz["tooltip_text"] = gdf_hex_pvz["pvz_count"].apply(lambda x: f"Всего пунктов логистики в соте: {x}")

st.subheader("📦 Пункты выдачи заказов (Ozon, WB, Яндекс, СДЭК)")
st.caption(f"Найдено ПВЗ и пунктов логистики: **{len(df_pvz)}**")
pvz_cols = st.columns([3, 1])
with pvz_cols[0]:
    pvz_map = build_pvz_map(CENTER_LAT, CENTER_LNG, RADIUS_METER, df_pvz, gdf_hex_pvz)
    st_folium(pvz_map, width=700, height=500, returned_objects=[])
with pvz_cols[1]:
    if not df_pvz.empty:
        brand_counts = df_pvz["brand"].value_counts()
        st.markdown("**Бренды:**")
        for b, c in brand_counts.items():
            st.markdown(f"- {b}: **{c}**")
    else:
        st.info("ПВЗ не найдены.")
    if len(df_pvz) > 10:
        st.success("✅ Развитая логистическая инфраструктура")
    elif len(df_pvz) > 3:
        st.info("ℹ️ Умеренное присутствие ПВЗ")
    else:
        st.warning("⚠️ Мало ПВЗ — возможно, спальный район")

st.divider()

# ═══════════════════════════════════════════════════════════════
# ИТОГОВАЯ СВОДКА (БЕЗ ИИ)
# ═══════════════════════════════════════════════════════════════
progress_bar.progress(100, text="✅ Анализ завершён!")

st.subheader("📊 Итоговая сводка по локации")

summary_data = {
    "Медицинских объектов": len(df_med),
    "Оценочное население": f"~{total_pop:,}",
    "Средний индекс платёжеспособности": avg_wealth,
    "Макс. пешеходный трафик": int(max_ped),
    "Макс. автомобильный трафик": int(max_auto),
    "ПВЗ в радиусе": len(df_pvz),
    "Премиальных ячеек": high_wealth_cells,
}

sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
metrics_list = list(summary_data.items())
for i, (label, value) in enumerate(metrics_list[:4]):
    with [sum_col1, sum_col2, sum_col3, sum_col4][i]:
        st.metric(label, value)

sum_col5, sum_col6, sum_col7 = st.columns(3)
for i, (label, value) in enumerate(metrics_list[4:]):
    with [sum_col5, sum_col6, sum_col7][i]:
        st.metric(label, value)

# Простая эвристическая оценка (без ИИ)
st.markdown("---")
st.markdown("### 🎯 Эвристический вердикт (без ИИ)")

score = 0
reasons = []
risks = []

# Плюсы
if total_pop > 30000:
    score += 2
    reasons.append("Высокая плотность населения — хороший поток пациентов")
elif total_pop > 15000:
    score += 1
    reasons.append("Средняя плотность населения")

if high_wealth_cells > 5:
    score += 2
    reasons.append("Есть премиальные ячейки с высокой платёжеспособностью")
elif avg_wealth > 50:
    score += 1
    reasons.append("Средний уровень платёжеспособности")

if max_ped > 80:
    score += 2
    reasons.append("Высокий пешеходный трафик — отличная проходимость")
elif max_ped > 40:
    score += 1
    reasons.append("Достаточный пешеходный трафик")

if max_auto > 40:
    score += 1
    reasons.append("Хорошая автомобильная доступность")

if len(df_pvz) > 8:
    score += 1
    reasons.append("Развитая логистика — активный район")

# Минусы / риски
if len(df_med) > 12:
    score -= 2
    risks.append("Высокая конкуренция — много медицинских объектов")
elif len(df_med) > 6:
    score -= 1
    risks.append("Умеренная конкуренция")

if total_pop < 10000:
    score -= 2
    risks.append("Низкая плотность населения — риск недозагрузки")

if max_ped < 20 and max_auto < 15:
    score -= 2
    risks.append("Слабый трафик — сложно привлечь пациентов")

if avg_wealth < 20:
    score -= 1
    risks.append("Низкая платёжеспособность — ограниченный спрос на платные услуги")

verdict_col, detail_col = st.columns([1, 2])
with verdict_col:
    if score >= 5:
        st.success("### ✅ РЕКОМЕНДУЕТСЯ\nЛокация сильная для открытия клиники.")
    elif score >= 2:
        st.warning("### ⚠️ ОТКРЫВАТЬ С ОСТОРОЖНОСТЬЮ\nЕсть потенциал, но важны нюансы.")
    else:
        st.error("### ❌ НЕ РЕКОМЕНДУЕТСЯ\nВысокие риски, слабые факторы успеха.")
    st.caption(f"Набрано баллов: {score}/8")

with detail_col:
    if reasons:
        st.markdown("**Главные плюсы:**")
        for r in reasons:
            st.markdown(f"- ✅ {r}")
    if risks:
        st.markdown("**Скрытые риски:**")
        for r in risks:
            st.markdown(f"- ⚠️ {r}")

st.markdown("---")
st.caption("💡 Данные получены из OpenStreetMap. Оценка населения приблизительная (на основе площади застройки и этажности).")
