# -*- coding: utf-8 -*-
"""
GeoMarketing AI — Clinic Location Сравнение v4.0
Без st.cache_data. Результаты хранятся в session_state.
"""

import importlib.util
import json
import math
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from openai import OpenAI
from pydantic import BaseModel, Field

# Graceful openpyxl handling
if importlib.util.find_spec("openpyxl") is None:
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "openpyxl", "--quiet", "--user"],
            check=False, capture_output=True, text=True
        )
    except Exception:
        pass

try:
    import openpyxl
except ImportError:
    openpyxl = None

# ==============================================================================
# STREAMLIT CONFIG
# ==============================================================================
st.set_page_config(
    page_title="Геомаркетинг клиники — Сравнение v4.0",
    page_icon="📍",
    layout="wide",
)

st.title("📍 Геомаркетинговый анализ локации клиники — v4.0")
st.caption("Загрузка референсов из файла + ручной чеклист target. Без ПроДокторов.")

# ==============================================================================
# КОНФИГУРАЦИЯ
# ==============================================================================
DEFAULT_MODEL = "gpt-5.1"
MODEL_REASONING = "low"

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

REQUEST_HEADERS = {
    "User-Agent": "ClinicGeoAnalytics/4.0 (streamlit-cloud; business use)"
}

BLOCK_WEIGHTS = {
    "location_params": 0.15,
    "parking_access":  0.20,
    "demand":          0.20,
    "competition":     0.15,
    "medical_eco":     0.15,
    "visibility_env":  0.15,
}

FACTORS = [
    ("location_param_score",    "location_params", 1.00, "rule",  "Базовые параметры локации"),
    ("parking_proximity",       "parking_access",  0.30, "osm",   "Близость парковки"),
    ("parking_supply",          "parking_access",  0.25, "osm",   "Ёмкость парковки"),
    ("vehicle_access",          "parking_access",  0.25, "osm",   "Удобство подъезда на авто"),
    ("public_transport",        "parking_access",  0.20, "osm",   "Общественный транспорт"),
    ("population_density",      "demand",          0.25, "osm",   "Плотность жилой застройки"),
    ("income_fit",              "demand",          0.20, "ai",    "Соответствие доходов ЦА"),
    ("age_fit",                 "demand",          0.15, "ai",    "Возрастное соответствие ЦА"),
    ("gender_fit",              "demand",          0.15, "ai",    "Половое соответствие ЦА"),
    ("family_profile",          "demand",          0.15, "ai",    "Семейный профиль района"),
    ("daytime_balance",         "demand",          0.10, "ai",    "Баланс дневного/жилого населения"),
    ("competitor_density",      "competition",     0.40, "osm",   "Плотность конкурентов"),
    ("competitor_strength",     "competition",     0.35, "ai",    "Сила конкурентов"),
    ("market_gap",              "competition",     0.25, "ai",    "Незакрытый спрос / рыночная ниша"),
    ("hospital_synergy",        "medical_eco",     0.50, "osm",   "Близость к государственным больницам (ОМС)"),
    ("medical_cluster",         "medical_eco",     0.50, "osm",   "Медицинский кластер"),
    ("visibility",              "visibility_env",  0.35, "osm",   "Видимость с дороги"),
    ("road_type_fit",           "visibility_env",  0.25, "osm",   "Тип трафика (жилой vs офисный)"),
    ("pedestrian_comfort",      "visibility_env",  0.20, "osm",   "Пешеходный комфорт"),
    ("noise_safety",            "visibility_env",  0.20, "ai",    "Шум и безопасность"),
]

FACTOR_KEYS = [f[0] for f in FACTORS]
FACTOR_BLOCK = {f[0]: f[1] for f in FACTORS}
FACTOR_WEIGHT_IN_BLOCK = {f[0]: f[2] for f in FACTORS}
FACTOR_SOURCE = {f[0]: f[3] for f in FACTORS}
FACTOR_LABEL = {f[0]: f[4] for f in FACTORS}
LOW_IS_BAD = {"competitor_density"}
FACTOR_GLOBAL_WEIGHT = {f[0]: f[2] * BLOCK_WEIGHTS[f[1]] for f in FACTORS}
FACTOR_DESCRIPTIONS = {
    "location_param_score": "Базовые параметры самого помещения. 100 = идеально: 1 этаж, отдельный вход, видимость с улицы, первая линия. Ниже 50 = серьёзные проблемы с доступностью.",
    "parking_proximity": "Насколько близко можно припарковаться к клинике. 90+ = парковка прямо у входа. Ниже 30 = искать место далеко, пациенты уходят к конкурентам.",
    "parking_supply": "Достаточно ли парковочных мест в целом. 90+ = много мест, свободно всегда. Ниже 25 = дефицит, особенно в часы пик.",
    "vehicle_access": "Удобство подъезда на машине. 90+ = широкая магистраль с удобным съездом. Ниже 30 = глухой переулок, пробки, сложный разворот.",
    "public_transport": "Доступность остановок и маршрутов. 90+ = метро/трамвай + автобусы в 150 м. Ниже 20 = далеко от всего, только на машине.",
    "population_density": "Плотность жилой застройки вокруг. 90+ = плотные многоэтажные кварталы. Ниже 30 = частный сектор, промзона, пустыри.",
    "income_fit": "Соответствие доходов жителей среднему чеку клиники. 80+ = жители платят ваш чек без проблем. Ниже 40 = район бедный, чек завышен.",
    "age_fit": "Возраст жителей соответствует вашей ЦА. 80+ = много семей с детьми и людей 30–60 лет. Ниже 40 = студенты или пенсионеры.",
    "gender_fit": "Половой состав соответствует профилю клиники. 80+ = женщины (если клиника женская) или равномерно. Ниже 40 = мужской район при женской клинике.",
    "family_profile": "Семейный состав района. 80+ = много семей с детьми (педиатрия, вакцинация). Ниже 40 = одиночки, пары без детей.",
    "daytime_balance": "Баланс жителей и офисных работников. 80+ = жилой район, люди дома вечерами и выходными. Ниже 30 = офисный район, пустой вечером.",
    "competitor_density": "Сколько клинок вокруг (в радиусе 2 км). ВНИМАНИЕ: 100 = очень много конкурентов (ПЛОХО). 0 = нет конкурентов (хорошо, если есть спрос).",
    "competitor_strength": "Насколько сильны конкуренты. 100 = слабые/нет сетевых, можно выиграть. 0 = сильные федеральные сети (СМ-Клиника, Медси и т.д.).",
    "market_gap": "Незакрытый спрос на услуги клиники. 80+ = много жителей, мало клиник, люди ездят в другой район. Ниже 30 = рынок перенасыщен.",
    "hospital_synergy": "Близость государственных больниц и поликлиник (в радиусе 2 км). 90+ = крупная поликлиника/больница рядом. Источник пациентов из ОМС. Ниже 30 = нет гос. медицины поблизости.",
    "medical_cluster": "Концентрация медучреждений в радиусе 2 км. 80+ = медицинский квартал (люди уже едут сюда лечиться). Ниже 30 = медицина разрознена.",
    "visibility": "Насколько хорошо клинику видно с дороги. 90+ = витрина на главной магистрали. Ниже 30 = двор, подвал, за углом.",
    "road_type_fit": "Тип проезжающего/проходящего трафика. 85+ = жилой район, ваши пациенты живут рядом. Ниже 30 = трасса/промзона/офисный трафик, который едет домой в другой район.",
    "pedestrian_comfort": "Удобство для пешеходов. 80+ = широкие тротуары, освещение, озеленение. Ниже 35 = нет тротуаров, грязь, небезопасно.",
    "noise_safety": "Шум и безопасность района. 90+ = тихий спальный район, безопасно вечером. Ниже 30 = шумная магистраль, промзона, небезопасно.",
    "traffic_quality": "Качество трафика (не количество, а ЦА). 80+ = мимо идут/едут ваши потенциальные пациенты. Ниже 30 = трафик нецелевой (грузовики, туристы, студенты).",
}


# ==============================================================================
# PYDANTIC
# ==============================================================================
class GeoAIFullProfile(BaseModel):
    parking_proximity: int = Field(ge=0, le=100)
    parking_supply: int = Field(ge=0, le=100)
    vehicle_access: int = Field(ge=0, le=100)
    public_transport: int = Field(ge=0, le=100)
    population_density: int = Field(ge=0, le=100)
    competitor_density: int = Field(ge=0, le=100)
    hospital_synergy: int = Field(ge=0, le=100)
    medical_cluster: int = Field(ge=0, le=100)
    visibility: int = Field(ge=0, le=100)
    road_type_fit: int = Field(ge=0, le=100)
    pedestrian_comfort: int = Field(ge=0, le=100)
    income_fit: int = Field(ge=0, le=100)
    age_fit: int = Field(ge=0, le=100)
    gender_fit: int = Field(ge=0, le=100)
    family_profile: int = Field(ge=0, le=100)
    daytime_balance: int = Field(ge=0, le=100)
    competitor_strength: int = Field(ge=0, le=100)
    market_gap: int = Field(ge=0, le=100)
    noise_safety: int = Field(ge=0, le=100)
    traffic_quality: int = Field(ge=0, le=100)
    profile_confidence: int = Field(ge=0, le=100)
    evidence_quality: int = Field(ge=0, le=100)


class GeoProfileItemFull(BaseModel):
    key: str
    profile: GeoAIFullProfile


class GeoProfileBatchFull(BaseModel):
    profiles: List[GeoProfileItemFull]


# ==============================================================================
# OPENAI
# ==============================================================================
def call_batch_ai_full(client: OpenAI, model: str, system_prompt: str, user_prompt: str) -> Optional[GeoProfileBatchFull]:
    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=GeoProfileBatchFull,
        timeout=120,
    )
    if model.startswith("gpt-5"):
        kwargs["reasoning_effort"] = MODEL_REASONING
    response = client.beta.chat.completions.parse(**kwargs)
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise ValueError("OpenAI не вернул структурированный ответ.")
    return parsed


def make_default_full_profile() -> dict:
    return {
        "parking_proximity": 50, "parking_supply": 50, "vehicle_access": 50,
        "public_transport": 50, "population_density": 50, "competitor_density": 50,
        "hospital_synergy": 50, "medical_cluster": 50,
        "visibility": 50, "road_type_fit": 50, "pedestrian_comfort": 50,
        "income_fit": 50, "age_fit": 50, "gender_fit": 50, "family_profile": 50,
        "daytime_balance": 50, "competitor_strength": 50, "market_gap": 50,
        "noise_safety": 50, "traffic_quality": 50, "profile_confidence": 30, "evidence_quality": 25,
    }


# ==============================================================================
# ГЕОКОДИРОВАНИЕ
# ==============================================================================
def get_exact_coordinates(address: str) -> Tuple[Optional[float], Optional[float]]:
    try:
        response = requests.get(
            "https://geocode.xyz",
            params={"locate": address, "json": "1", "region": "RU"},
            timeout=10,
        )
        if response.status_code == 200:
            data = response.json()
            if "error" not in data and "latt" in data and "longt" in data:
                lat = float(data["latt"])
                lon = float(data["longt"])
                if abs(lat) > 0.01 and abs(lon) > 0.01:
                    return lat, lon
    except Exception:
        pass

    try:
        response = requests.get(
            "https://photon.komoot.io/api/",
            params={"q": address, "limit": 1},
            timeout=10,
        )
        if response.status_code == 200:
            data = response.json()
            features = data.get("features", [])
            if features and len(features) > 0:
                coords = features[0].get("geometry", {}).get("coordinates", [])
                if len(coords) >= 2:
                    lon, lat = coords[0], coords[1]
                    if abs(lat) > 0.01 and abs(lon) > 0.01:
                        return lat, lon
    except Exception:
        pass

    try:
        response = requests.get(
            NOMINATIM_URL,
            params={"q": address, "format": "jsonv2", "limit": 1, "addressdetails": 1},
            headers=REQUEST_HEADERS,
            timeout=10,
        )
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0:
                return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception:
        pass

    return None, None


# ==============================================================================
# OVERPASS / OSM
# ==============================================================================
OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]


def _overpass_request(query: str, timeout: int = 25) -> List[dict]:
    """Overpass с retry и fallback на зеркала."""
    last_error = None
    for url in OVERPASS_MIRRORS:
        for attempt in range(2):
            try:
                response = requests.post(url, data={"data": query}, headers=REQUEST_HEADERS, timeout=timeout)
                response.raise_for_status()
                return response.json().get("elements", [])
            except requests.exceptions.Timeout:
                last_error = f"timeout ({timeout}s)"
                if attempt == 0:
                    time.sleep(2)
                    continue
            except requests.exceptions.HTTPError as e:
                last_error = f"HTTP {e.response.status_code}"
                break  # не retry при 4xx/5xx
            except Exception as e:
                last_error = str(e)
                break
    st.warning(f"⚠️ OSM Overpass недоступен: {last_error}. Все факторы будут оценены AI.")
    return []


def collect_osm_context(lat: float, lon: float) -> dict:
    # Два легких запроса вместо одного тяжелого
    query_medical = f"""
    [out:json][timeout:45];
    (
      nwr(around:2000,{lat},{lon})["amenity"~"hospital|clinic|doctors"];
      nwr(around:2000,{lat},{lon})["healthcare"~"centre|clinic|doctor"];
    );
    out center tags;
    """
    query_infra = f"""
    [out:json][timeout:45];
    (
      nwr(around:1000,{lat},{lon})["amenity"="parking"];
      nwr(around:300,{lat},{lon})["highway"~"bus_stop|platform"];
      nwr(around:800,{lat},{lon})["public_transport"];
      nwr(around:1000,{lat},{lon})["building"~"apartments|residential|house|detached|office|commercial|retail"];
      nwr(around:800,{lat},{lon})["highway"~"primary|secondary|tertiary|residential"];
    );
    out center tags;
    """
    query = query_medical
    elements = _overpass_request(query_medical, timeout=50)
    elements_infra = _overpass_request(query_infra, timeout=50)
    elements = elements + elements_infra

    if not elements:
        return {"available": False, "error": "empty_or_timeout", "counts": {}, "roads": {}, "landuse": {}, "buildings": {}, "raw_count": 0}

    counts = {
        "clinic_300m": 0, "clinic_800m": 0,
        "hospital_300m": 0, "hospital_800m": 0,
        "school_800m": 0,
        "parking_500m": 0, "parking_1000m": 0,
        "bus_stop_300m": 0, "public_transport_800m": 0,
        "residential_buildings_500m": 0, "residential_buildings_1000m": 0,
        "office_buildings_500m": 0, "office_buildings_1000m": 0,
        "primary_road_800m": 0, "secondary_road_800m": 0,
        "tertiary_road_800m": 0, "residential_road_800m": 0,
    }
    roads = {}
    landuse = {}
    buildings = {}

    for el in elements:
        tags = el.get("tags", {})
        amenity = tags.get("amenity")
        highway = tags.get("highway")
        building = tags.get("building")
        land = tags.get("landuse")
        healthcare = tags.get("healthcare")

        if amenity in ("clinic", "doctors") or healthcare in ("clinic", "doctor", "centre"):
            counts["clinic_800m"] += 1
        if amenity == "hospital" or healthcare == "hospital":
            counts["hospital_800m"] += 1
        if amenity in ("school", "kindergarten"):
            counts["school_800m"] += 1
        if amenity == "parking":
            counts["parking_1000m"] += 1
        if highway in ("bus_stop", "platform"):
            counts["bus_stop_300m"] += 1
        if "public_transport" in tags:
            counts["public_transport_800m"] += 1
        if building in ("apartments", "residential", "house", "detached"):
            counts["residential_buildings_1000m"] += 1
        if building in ("office", "commercial", "retail"):
            counts["office_buildings_1000m"] += 1
        if highway:
            roads[highway] = roads.get(highway, 0) + 1
            if highway == "primary":
                counts["primary_road_800m"] += 1
            elif highway == "secondary":
                counts["secondary_road_800m"] += 1
            elif highway == "tertiary":
                counts["tertiary_road_800m"] += 1
            elif highway == "residential":
                counts["residential_road_800m"] += 1
        if land:
            landuse[land] = landuse.get(land, 0) + 1
        if building:
            buildings[building] = buildings.get(building, 0) + 1

    counts["clinic_300m"] = counts["clinic_800m"]
    counts["hospital_300m"] = counts["hospital_800m"]
    counts["parking_500m"] = counts["parking_1000m"]
    counts["residential_buildings_500m"] = counts["residential_buildings_1000m"]
    counts["office_buildings_500m"] = counts["office_buildings_1000m"]

    return {
        "available": True,
        "error": None,
        "counts": counts,
        "roads": roads,
        "landuse": landuse,
        "buildings": buildings,
        "raw_count": len(elements),
    }


# ==============================================================================
# OSM → SCORE
# ==============================================================================
def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def osm_to_factor_scores(osm: dict) -> Dict[str, float]:
    if not osm.get("available"):
        return {
            "parking_proximity": 50, "parking_supply": 50, "vehicle_access": 50,
            "public_transport": 50, "population_density": 50, "competitor_density": 50,
            "pharmacy_synergy": 50, "diagnostics_synergy": 50, "hospital_synergy": 50,
            "medical_cluster": 50, "visibility": 50, "road_type_fit": 50, "pedestrian_comfort": 50,
        }

    c = osm.get("counts", {})
    scores = {}

    parking_500 = c.get("parking_500m", 0)
    parking_1000 = c.get("parking_1000m", 0)
    if parking_500 >= 3:
        scores["parking_proximity"] = 95
    elif parking_500 >= 1:
        scores["parking_proximity"] = 70
    elif parking_1000 >= 3:
        scores["parking_proximity"] = 50
    elif parking_1000 >= 1:
        scores["parking_proximity"] = 30
    else:
        scores["parking_proximity"] = 5

    if parking_1000 >= 10:
        scores["parking_supply"] = 95
    elif parking_1000 >= 5:
        scores["parking_supply"] = 75
    elif parking_1000 >= 2:
        scores["parking_supply"] = 50
    elif parking_1000 >= 1:
        scores["parking_supply"] = 25
    else:
        scores["parking_supply"] = 0

    primary = c.get("primary_road_800m", 0)
    secondary = c.get("secondary_road_800m", 0)
    tertiary = c.get("tertiary_road_800m", 0)
    residential = c.get("residential_road_800m", 0)
    if primary > 0 and (secondary > 0 or tertiary > 0):
        scores["vehicle_access"] = 90
    elif secondary > 0 and tertiary > 0:
        scores["vehicle_access"] = 80
    elif tertiary > 0 and residential > 0:
        scores["vehicle_access"] = 70
    elif residential > 0:
        scores["vehicle_access"] = 60
    elif tertiary > 0:
        scores["vehicle_access"] = 50
    else:
        scores["vehicle_access"] = 30

    bus = c.get("bus_stop_300m", 0)
    pt = c.get("public_transport_800m", 0)
    if bus >= 2 and pt >= 3:
        scores["public_transport"] = 95
    elif bus >= 1 and pt >= 2:
        scores["public_transport"] = 75
    elif bus >= 1 or pt >= 2:
        scores["public_transport"] = 55
    elif pt >= 1:
        scores["public_transport"] = 35
    else:
        scores["public_transport"] = 15

    res_500 = c.get("residential_buildings_500m", 0)
    res_1000 = c.get("residential_buildings_1000m", 0)
    if res_500 >= 30:
        scores["population_density"] = 95
    elif res_500 >= 15:
        scores["population_density"] = 80
    elif res_500 >= 8:
        scores["population_density"] = 65
    elif res_1000 >= 20:
        scores["population_density"] = 50
    elif res_1000 >= 10:
        scores["population_density"] = 35
    else:
        scores["population_density"] = 20

    clinic_2km = c.get("clinic_800m", 0)  # теперь это 2км
    if clinic_2km >= 10:
        scores["competitor_density"] = 95
    elif clinic_2km >= 6:
        scores["competitor_density"] = 75
    elif clinic_2km >= 3:
        scores["competitor_density"] = 50
    elif clinic_2km >= 1:
        scores["competitor_density"] = 25
    else:
        scores["competitor_density"] = 5

    hosp_2km = c.get("hospital_800m", 0)  # теперь это 2км
    if hosp_2km >= 3:
        scores["hospital_synergy"] = 95
    elif hosp_2km >= 1:
        scores["hospital_synergy"] = 75
    else:
        scores["hospital_synergy"] = 30

    med_total = c.get("clinic_800m", 0) + hosp_2km  # оба теперь 2км
    if med_total >= 15:
        scores["medical_cluster"] = 95
    elif med_total >= 8:
        scores["medical_cluster"] = 80
    elif med_total >= 4:
        scores["medical_cluster"] = 60
    elif med_total >= 2:
        scores["medical_cluster"] = 40
    else:
        scores["medical_cluster"] = 20

    if primary > 0:
        scores["visibility"] = 90
    elif secondary > 0:
        scores["visibility"] = 80
    elif tertiary > 0:
        scores["visibility"] = 60
    elif residential > 0:
        scores["visibility"] = 40
    else:
        scores["visibility"] = 20

    office_b = c.get("office_buildings_500m", 0)
    if residential > 0 and (primary > 0 or secondary > 0):
        scores["road_type_fit"] = 85
    elif residential > 0 and tertiary > 0:
        scores["road_type_fit"] = 75
    elif tertiary > 0 and office_b < 5:
        scores["road_type_fit"] = 60
    elif office_b >= 10:
        scores["road_type_fit"] = 30
    else:
        scores["road_type_fit"] = 50

    if residential > 5:
        scores["pedestrian_comfort"] = 80
    elif residential > 2:
        scores["pedestrian_comfort"] = 65
    else:
        scores["pedestrian_comfort"] = 40

    return scores


# ==============================================================================
# ПАРАМЕТРЫ ЛОКАЦИИ
# ==============================================================================

# ==============================================================================
# USER OVERRIDES (приоритет пользовательских данных над AI/OSM)
# ==============================================================================
def apply_user_overrides(profile: dict, known_data: dict) -> dict:
    """
    Если пользователь ввёл конкретные данные о районе — они имеют
    ПРИОРИТЕТ над AI и OSM. Возвращает словарь {factor: (old_val, new_val, reason)}.
    """
    overrides = {}

    # --- ПАРКОВКА ---
    parking = str(known_data.get("parking", "")).lower().strip()
    if parking == "нет":
        old = profile.get("parking_proximity", 50)
        profile["parking_proximity"] = 8
        overrides["parking_proximity"] = (old, 8, "Пользователь: парковки нет")
        old = profile.get("parking_supply", 50)
        profile["parking_supply"] = 5
        overrides["parking_supply"] = (old, 5, "Пользователь: парковки нет")
    elif parking == "ограничена":
        old = profile.get("parking_proximity", 50)
        profile["parking_proximity"] = 35
        overrides["parking_proximity"] = (old, 35, "Пользователь: парковка ограничена")
        old = profile.get("parking_supply", 50)
        profile["parking_supply"] = 25
        overrides["parking_supply"] = (old, 25, "Пользователь: парковка ограничена")
    elif parking == "да":
        old = profile.get("parking_proximity", 50)
        profile["parking_proximity"] = 85
        overrides["parking_proximity"] = (old, 85, "Пользователь: парковка есть")
        old = profile.get("parking_supply", 50)
        profile["parking_supply"] = 75
        overrides["parking_supply"] = (old, 75, "Пользователь: парковка есть")

    # --- АВТОТРАФИК ---
    traffic_car = str(known_data.get("traffic_car", "")).lower().strip()
    if traffic_car == "экстремально_high":
        old = profile.get("vehicle_access", 50)
        profile["vehicle_access"] = 35
        overrides["vehicle_access"] = (old, 35, "Пользователь: экстремально высокий автотрафик (пробки)")
        old = profile.get("road_type_fit", 50)
        profile["road_type_fit"] = 75
        overrides["road_type_fit"] = (old, 75, "Пользователь: высокий автотрафик")
    elif traffic_car == "высокий":
        old = profile.get("vehicle_access", 50)
        profile["vehicle_access"] = 65
        overrides["vehicle_access"] = (old, 65, "Пользователь: высокий автотрафик")
        old = profile.get("road_type_fit", 50)
        profile["road_type_fit"] = 80
        overrides["road_type_fit"] = (old, 80, "Пользователь: высокий автотрафик")
    elif traffic_car == "средний":
        old = profile.get("vehicle_access", 50)
        profile["vehicle_access"] = 55
        overrides["vehicle_access"] = (old, 55, "Пользователь: средний автотрафик")
    elif traffic_car == "низкий":
        old = profile.get("vehicle_access", 50)
        profile["vehicle_access"] = 25
        overrides["vehicle_access"] = (old, 25, "Пользователь: низкий автотрафик")

    # --- ПЕШИЙ ТРАФИК ---
    traffic_ped = str(known_data.get("traffic_ped", "")).lower().strip()
    if traffic_ped == "высокий":
        old = profile.get("pedestrian_comfort", 50)
        profile["pedestrian_comfort"] = 85
        overrides["pedestrian_comfort"] = (old, 85, "Пользователь: высокий пешеходный трафик")
        old = profile.get("visibility", 50)
        profile["visibility"] = 80
        overrides["visibility"] = (old, 80, "Пользователь: высокий пешеходный трафик")
    elif traffic_ped == "средний":
        old = profile.get("pedestrian_comfort", 50)
        profile["pedestrian_comfort"] = 65
        overrides["pedestrian_comfort"] = (old, 65, "Пользователь: средний пешеходный трафик")
    elif traffic_ped == "низкий":
        old = profile.get("pedestrian_comfort", 50)
        profile["pedestrian_comfort"] = 35
        overrides["pedestrian_comfort"] = (old, 35, "Пользователь: низкий пешеходный трафик")
        old = profile.get("visibility", 50)
        profile["visibility"] = 45
        overrides["visibility"] = (old, 45, "Пользователь: низкий пешеходный трафик")

    # --- ПЛОТНОСТЬ НАСЕЛЕНИЯ ---
    pop_dens = str(known_data.get("population_density", "")).lower().strip()
    if pop_dens == "очень_high":
        old = profile.get("population_density", 50)
        profile["population_density"] = 95
        overrides["population_density"] = (old, 95, "Пользователь: очень высокая плотность")
    elif pop_dens == "высокая":
        old = profile.get("population_density", 50)
        profile["population_density"] = 80
        overrides["population_density"] = (old, 80, "Пользователь: высокая плотность")
    elif pop_dens == "средняя":
        old = profile.get("population_density", 50)
        profile["population_density"] = 55
        overrides["population_density"] = (old, 55, "Пользователь: средняя плотность")
    elif pop_dens == "ниже_medium":
        old = profile.get("population_density", 50)
        profile["population_density"] = 35
        overrides["population_density"] = (old, 35, "Пользователь: ниже средней плотность")
    elif pop_dens == "низкая":
        old = profile.get("population_density", 50)
        profile["population_density"] = 15
        overrides["population_density"] = (old, 15, "Пользователь: низкая плотность")

    # --- ТРАНСПОРТНАЯ ДОСТУПНОСТЬ ---
    trans = str(known_data.get("transport_access", "")).lower().strip()
    if trans == "отличная":
        old = profile.get("public_transport", 50)
        profile["public_transport"] = 95
        overrides["public_transport"] = (old, 95, "Пользователь: отличная транспортная доступность")
    elif trans == "очень_good":
        old = profile.get("public_transport", 50)
        profile["public_transport"] = 85
        overrides["public_transport"] = (old, 85, "Пользователь: очень хорошая транспортная доступность")
    elif trans == "хорошая":
        old = profile.get("public_transport", 50)
        profile["public_transport"] = 70
        overrides["public_transport"] = (old, 70, "Пользователь: хорошая транспортная доступность")
    elif trans == "средняя":
        old = profile.get("public_transport", 50)
        profile["public_transport"] = 50
        overrides["public_transport"] = (old, 50, "Пользователь: средняя транспортная доступность")

    # --- КОНКУРЕНТЫ (количество) ---
    comp_count = known_data.get("competitors_count")
    if comp_count is not None:
        try:
            comp_count = int(comp_count)
            old = profile.get("competitor_density", 50)
            if comp_count >= 10:
                profile["competitor_density"] = 95
                overrides["competitor_density"] = (old, 95, f"Пользователь: {comp_count} конкурентов")
            elif comp_count >= 6:
                profile["competitor_density"] = 75
                overrides["competitor_density"] = (old, 75, f"Пользователь: {comp_count} конкурентов")
            elif comp_count >= 3:
                profile["competitor_density"] = 50
                overrides["competitor_density"] = (old, 50, f"Пользователь: {comp_count} конкурентов")
            elif comp_count >= 1:
                profile["competitor_density"] = 25
                overrides["competitor_density"] = (old, 25, f"Пользователь: {comp_count} конкурентов")
            else:
                profile["competitor_density"] = 5
                overrides["competitor_density"] = (old, 5, "Пользователь: нет конкурентов")
        except (ValueError, TypeError):
            pass

    # --- ТРАВМПУНКТ ---
    trauma = str(known_data.get("has_trauma_center", "")).lower().strip()
    if trauma == "да":
        old = profile.get("hospital_synergy", 50)
        profile["hospital_synergy"] = 85
        overrides["hospital_synergy"] = (old, 85, "Пользователь: травмпункт рядом")
    elif trauma == "нет":
        old = profile.get("hospital_synergy", 50)
        profile["hospital_synergy"] = 20
        overrides["hospital_synergy"] = (old, 20, "Пользователь: травмпункта нет")

    # --- ЦЕНОВОЙ СЕГМЕНТ → income_fit ---
    price_seg = str(known_data.get("price_segment", "")).lower().strip()
    if price_seg in ("эконом", "средний"):
        old = profile.get("income_fit", 50)
        profile["income_fit"] = 55
        overrides["income_fit"] = (old, 55, f"Пользователь: ценовой сегмент {price_seg}")
    elif price_seg in ("средний+", "бизнес"):
        old = profile.get("income_fit", 50)
        profile["income_fit"] = 75
        overrides["income_fit"] = (old, 75, f"Пользователь: ценовой сегмент {price_seg}")
    elif price_seg == "премиум":
        old = profile.get("income_fit", 50)
        profile["income_fit"] = 90
        overrides["income_fit"] = (old, 90, "Пользователь: премиум сегмент")

    # --- ТИП ЗАСТРОЙКИ 1км → road_type_fit, daytime_balance ---
    btype = str(known_data.get("building_type_1km", "")).lower()
    if "бц" in btype or "офис" in btype or "делов" in btype:
        old = profile.get("road_type_fit", 50)
        profile["road_type_fit"] = 40
        overrides["road_type_fit"] = (old, 40, "Пользователь: преимущественно офисная застройка")
        old = profile.get("daytime_balance", 50)
        profile["daytime_balance"] = 35
        overrides["daytime_balance"] = (old, 35, "Пользователь: офисный район (пустой вечером)")
    elif "жил" in btype or "многоэтаж" in btype or "квартал" in btype:
        old = profile.get("road_type_fit", 50)
        profile["road_type_fit"] = 85
        overrides["road_type_fit"] = (old, 85, "Пользователь: жилая застройка")
        old = profile.get("daytime_balance", 50)
        profile["daytime_balance"] = 80
        overrides["daytime_balance"] = (old, 80, "Пользователь: жилой район")
    elif "частн" in btype or "коттедж" in btype:
        old = profile.get("population_density", 50)
        profile["population_density"] = 30
        overrides["population_density"] = (old, 30, "Пользователь: частный сектор")

    return overrides


def compute_location_param_score(params: dict) -> Tuple[float, List[Tuple[str, int, str]]]:
    base = 100.0
    applied = []
    if params.get("floor") == "upper":
        base -= 25
        applied.append(("2+ этаж", 25, "Нет видимости, сложная навигация"))
    if not params.get("separate_entrance", True):
        base -= 20
        applied.append(("Нет отдельного входа", 20, "Барьер для пациентов"))
    if not params.get("street_visibility", True):
        base -= 15
        applied.append(("Нет видимости с улицы", 15, "Пациент не видит клинику"))
    if params.get("building_type") == "bc":
        base -= 15
        applied.append(("Бизнес-центр", 15, "Офисный трафик, выходные пустые"))
    elif params.get("building_type") == "mall":
        base -= 10
        applied.append(("Торговый центр", 10, "Нецелевой трафик, сложная навигация"))
    if not params.get("first_line", True):
        base -= 10
        applied.append(("Не первая линия", 10, "Меньше проходного трафика"))
    return clamp(base, 5.0), applied


def parse_benchmark_params(row: pd.Series) -> dict:
    bt_raw = str(row.get("building_type", "")).lower()
    if "жилой" in bt_raw:
        bt = "residential"
    elif "отдельное" in bt_raw or "особняк" in bt_raw:
        bt = "standalone"
    elif "бизнес" in bt_raw or "бц" in bt_raw:
        bt = "bc"
    elif "торгов" in bt_raw or "молл" in bt_raw or "пассаж" in bt_raw:
        bt = "mall"
    else:
        bt = "other"

    sv_raw = str(row.get("street_visibility", "")).lower()
    sv = any(x in sv_raw for x in ["отличн", "хорош", "первая линия"])

    fl_raw = str(row.get("first_line", "")).lower()
    first = "да" in fl_raw or "yes" in fl_raw

    return {
        "building_type": bt,
        "floor": "ground",
        "separate_entrance": True,
        "street_visibility": sv,
        "first_line": first,
    }


# ==============================================================================
# AI PROMPTS
# ==============================================================================
def build_ai_full_system_prompt() -> str:
    return """Ты — geo-marketing analyst для частных медицинских клиник формата «клиника у дома».

Оцени ВСЕ факторы по шкале 0–100:
1. parking_proximity — близость парковки (0 = нет в 1 км, 100 = у входа).
2. parking_supply — ёмкость парковки.
3. vehicle_access — удобство подъезда на авто.
4. public_transport — общественный транспорт.
5. population_density — плотность жилой застройки.
6. competitor_density — плотность конкурентов (100 = много конкурентов, это ПЛОХО).
7. hospital_synergy — близость к государственным больницам и поликлиникам (0 = нет гос. медучреждений, 100 = крупная больница в 300 м). Важно для перенаправления пациентов из ОМС.
8. medical_cluster — медицинский кластер (0 = нет медучреждений, 100 = медицинский квартал).
9. visibility — видимость с дороги (0 = двор/подвал, 100 = витрина на главной магистрали).
10. road_type_fit — тип трафика (0 = промзона/трасса, 100 = жилой район с трафиком целевой аудитории).
11. pedestrian_comfort — пешеходный комфорт (0 = нет тротуаров, 100 = широкие тротуары, озеленение).
12. income_fit — соответствие дохода населения среднему чеку клиники.
13. age_fit — возрастное соответствие целевой аудитории.
14. gender_fit — половое соответствие ЦА.
15. family_profile — семейный профиль района.
16. daytime_balance — баланс дневного/жилого населения.
17. competitor_strength — сила конкурентов (100 = слабые/отсутствуют).
18. market_gap — незакрытый спрос / рыночная ниша (100 = большой незакрытый спрос на услуги клиники).
19. noise_safety — шум и безопасность (100 = тихо и безопасно).
20. traffic_quality — качество трафика для ЦА (не количество, а соответствие целевой аудитории).

ПРАВИЛА:
- Используй переданные аудиторные данные и адрес.
- Если данных мало — снижай confidence и evidence_quality.
- Все оценки целые числа 0–100.
- profile_confidence и evidence_quality тоже 0–100.
"""


def safe_str(val) -> str:
    """Безопасное преобразование значения DataFrame в строку."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return ""
    return str(val)


def build_benchmark_batch_prompt(df: pd.DataFrame) -> str:
    chunks = []
    for idx, row in df.iterrows():
        chunk = f"""--- РЕФЕРЕНС {idx} ---
name: {safe_str(row.get('name', ''))}
address: {safe_str(row.get('address', ''))}
building_type: {safe_str(row.get('building_type', ''))}, storeys: {safe_str(row.get('number_of_storeys', ''))}
street_visibility: {safe_str(row.get('street_visibility', ''))}
first_line: {safe_str(row.get('first_line', ''))}
parking: {safe_str(row.get('parking', ''))}
traffic_car: {safe_str(row.get('traffic_car', ''))}
traffic_ped: {safe_str(row.get('traffic_ped', ''))}
population_density: {safe_str(row.get('population_density', ''))}
building_type_1km: {safe_str(row.get('building_type_1km', ''))}
avg_housing_price: {safe_str(row.get('avg_housing_price', ''))}
transport_stop_distance: {safe_str(row.get('transport_stop_distance', ''))}
transport_access: {safe_str(row.get('transport_access', ''))}
format: {safe_str(row.get('format', ''))}
result: {safe_str(row.get('result', ''))}
competitors_count: {safe_str(row.get('competitors_count', ''))}
competitors_list: {safe_str(row.get('competitors_list', ''))}
has_trauma_center: {safe_str(row.get('has_trauma_center', ''))}
price_segment: {safe_str(row.get('price_segment', ''))}
avg_ticket: {safe_str(row.get('avg_ticket', ''))}
year_opened: {safe_str(row.get('year_opened', ''))}
type: {safe_str(row.get('type', ''))}
"""
        chunks.append(chunk)

    return f"""Ты — geo-marketing analyst. Оцени гео-маркетинговый профиль для КАЖДОГО референса из списка ниже.
Для каждого референса оцени ВСЕ 21 фактор по шкале 0–100 на основе переданных аудиторных данных.
Используй свои знания о городе и районе по адресу.

{''.join(chunks)}

Верни РОВНО по одному профилю на каждый референс (key = индекс строки: 0,1,2...).
"""


def build_target_prompt(target_loc: dict, known_data: dict, osm: dict) -> str:
    p = target_loc.get("params", {})
    param_lines = []
    if p.get("building_type") == "bc": param_lines.append("Бизнес-центр")
    elif p.get("building_type") == "mall": param_lines.append("Торговый центр")
    elif p.get("building_type") == "residential": param_lines.append("Жилой дом")
    elif p.get("building_type") == "standalone": param_lines.append("Отдельное здание")
    else: param_lines.append("Другое")
    param_lines.append("2+ этаж" if p.get("floor") == "upper" else "1 этаж")
    param_lines.append("Отдельный вход" if p.get("separate_entrance") else "Нет отдельного входа")
    param_lines.append("Видимость с улицы" if p.get("street_visibility") else "Нет видимости")
    param_lines.append("Первая линия" if p.get("first_line") else "Не первая линия")

    known_lines = []
    for k, v in known_data.items():
        if v and str(v).lower() not in ("неизвестно", "", "nan", "none"):
            known_lines.append(f"{k}: {v}")

    osm_status = "✅ Доступен" if osm.get("available") else "❌ Недоступен"

    return f"""Оцени ВСЕ факторы для планируемой клиники:

Адрес: {target_loc["address"]}
Параметры объекта: {', '.join(param_lines)}
ЦА: возраст {target_loc['target_age']:.0f}; женщины {target_loc['share_female']*100:.0f}%; чек {target_loc['avg_ticket']:,} руб.

ИЗВЕСТНЫЕ ДАННЫЕ О РАЙОНЕ (подтверждены пользователем):
{'\n'.join(known_lines) if known_lines else 'Нет дополнительных данных.'}

OSM статус: {osm_status}
OSM counts: {json.dumps(osm.get('counts', {}), ensure_ascii=False)}

{'OSM НЕДОСТУПЕН. Оцени все факторы самостоятельно на основе адреса, параметров объекта и известных данных.' if not osm.get('available') else 'OSM доступен. Используй его данные для OSM-факторов, а остальные оцени самостоятельно.'}
"""


# ==============================================================================
# AI GENERATION (без cache_data)
# ==============================================================================
def generate_benchmark_profiles(api_key: str, model: str, df: pd.DataFrame) -> dict:
    """Генерирует профили для всех референсов батчем."""
    client = OpenAI(api_key=api_key)
    batch = call_batch_ai_full(
        client=client,
        model=model.strip(),
        system_prompt=build_ai_full_system_prompt(),
        user_prompt=build_benchmark_batch_prompt(df),
    )
    if batch is None:
        raise ValueError("OpenAI вернул None.")
    result = {}
    for item in batch.profiles:
        try:
            idx = int(item.key)
        except ValueError:
            continue
        if idx < 0 or idx >= len(df):
            continue
        row = df.iloc[idx]
        profile = item.profile.model_dump()
        bench_params = parse_benchmark_params(row)
        loc_score, _ = compute_location_param_score(bench_params)
        profile["location_param_score"] = loc_score
        result[idx] = {
            "name": safe_str(row.get("name", f"Референс {idx}")),
            "address": safe_str(row.get("address", "")),
            "status": safe_str(row.get("result", "")).lower().strip(),
            "profile": profile,
            "params": bench_params,
        }
    return result


def generate_target_profile(api_key: str, model: str, target_loc: dict, known_data: dict, osm: dict) -> dict:
    """AI оценивает target (все 21 фактор)."""
    client = OpenAI(api_key=api_key)
    batch = call_batch_ai_full(
        client=client,
        model=model.strip(),
        system_prompt=build_ai_full_system_prompt(),
        user_prompt=build_target_prompt(target_loc, known_data, osm),
    )
    if batch is None or not batch.profiles:
        raise ValueError("OpenAI не вернул профиль.")
    return batch.profiles[0].profile.model_dump()


# ==============================================================================
# SCORING ENGINE
# ==============================================================================
def compute_block_scores(full_profile: dict) -> Dict[str, float]:
    blocks = {b: [] for b in BLOCK_WEIGHTS}
    for factor in FACTOR_KEYS:
        block = FACTOR_BLOCK.get(factor, "")
        weight = FACTOR_WEIGHT_IN_BLOCK.get(factor, 0)
        value = full_profile.get(factor, 0)
        if factor in LOW_IS_BAD:
            value = 100.0 - value
        blocks[block].append((value, weight))
    result = {}
    for block, items in blocks.items():
        total_w = sum(w for _, w in items)
        if total_w <= 0:
            result[block] = 0.0
        else:
            result[block] = round(sum(v * w for v, w in items) / total_w, 1)
    return result


def compute_absolute_score(block_scores: Dict[str, float]) -> float:
    return round(sum(block_scores.get(b, 0) * BLOCK_WEIGHTS.get(b, 0) for b in BLOCK_WEIGHTS), 1)


def _norm_value(profile: dict, factor: str) -> float:
    v = profile.get(factor, 0)
    if factor in LOW_IS_BAD:
        v = 100.0 - v
    return v


def profile_vector(full_profile: dict) -> np.ndarray:
    return np.array([_norm_value(full_profile, f) for f in FACTOR_KEYS], dtype=float)


def схожесть_to_reference(target: dict, reference: dict) -> float:
    a = profile_vector(target)
    b = profile_vector(reference)
    weights = np.array([FACTOR_GLOBAL_WEIGHT.get(f, 0) for f in FACTOR_KEYS], dtype=float)
    total_w = np.sum(weights)
    if total_w <= 0:
        return 0.0
    distance = np.sum(np.abs(a - b) * weights) / total_w
    return round(clamp(100.0 - distance), 1)


def схожесть_debug(target: dict, reference: dict, ref_name: str) -> Tuple[float, List[Tuple[str, float, float, float]]]:
    a = profile_vector(target)
    b = profile_vector(reference)
    weights = np.array([FACTOR_GLOBAL_WEIGHT.get(f, 0) for f in FACTOR_KEYS], dtype=float)
    total_w = np.sum(weights)
    items = []
    for i, factor in enumerate(FACTOR_KEYS):
        diff = abs(a[i] - b[i])
        contrib = round(diff * weights[i] / total_w, 2) if total_w > 0 else 0.0
        items.append((FACTOR_LABEL.get(factor, factor), round(a[i], 1), round(b[i], 1), contrib))
    items.sort(key=lambda x: x[3], reverse=True)
    distance = np.sum(np.abs(a - b) * weights) / total_w if total_w > 0 else 100.0
    sim = round(clamp(100.0 - distance), 1)
    return sim, items


def group_centroid(profiles: List[dict]) -> dict:
    if not profiles:
        return {}
    centroid = {}
    for factor in FACTOR_KEYS:
        vals = [_norm_value(p, factor) for p in profiles]
        centroid[factor] = float(np.mean(vals))
    return centroid


def benchmark_analysis(target_profile: dict, benchmark_rows: List[dict]) -> dict:
    successful = [r for r in benchmark_rows if r.get("status") in ("успешная", "успешный", "successful")]
    weak = [r for r in benchmark_rows if r.get("status") in ("слабая", "слабый", "weak", "неуспешная", "неуспешный")]

    success_similarity = []
    for r in successful:
        sim, debug = схожесть_debug(target_profile, r.get("profile", {}), r.get("address", ""))
        success_similarity.append((r.get("address", ""), sim, debug, "успешная"))

    weak_similarity = []
    for r in weak:
        sim, debug = схожесть_debug(target_profile, r.get("profile", {}), r.get("address", ""))
        weak_similarity.append((r.get("address", ""), sim, debug, "слабая"))

    all_similarity = success_similarity + weak_similarity
    all_similarity.sort(key=lambda x: x[1], reverse=True)
    success_similarity.sort(key=lambda x: x[1], reverse=True)
    weak_similarity.sort(key=lambda x: x[1], reverse=True)

    successful_centroid = group_centroid([r.get("profile", {}) for r in successful])
    weak_centroid = group_centroid([r.get("profile", {}) for r in weak])

    to_success = схожесть_to_reference(target_profile, successful_centroid) if successful_centroid else 0.0
    to_weak = схожесть_to_reference(target_profile, weak_centroid) if weak_centroid else 0.0

    return {
        "all_similarity": [(a, b, c) for a, b, _, c in all_similarity],
        "all_debug": all_similarity,
        "success_similarity": [(a, b) for a, b, _, _ in success_similarity],
        "weak_similarity": [(a, b) for a, b, _, _ in weak_similarity],
        "success_debug": success_similarity,
        "weak_debug": weak_similarity,
        "successful_centroid_similarity": to_success,
        "weak_centroid_similarity": to_weak,
        "benchmark_gap": round(to_success - to_weak, 1),
    }


# ==============================================================================
# HARD RULES
# ==============================================================================
def calculate_hard_barriers(full_profile: dict, osm: dict, params: dict) -> List[str]:
    barriers = []
    c = osm.get("counts", {})
    osm_available = osm.get("available", False)

    if params.get("floor") == "upper":
        if params.get("building_type") in ("bc", "mall"):
            barriers.append("🚨 КРИТИЧНО: Объект на 2+ этаже в БЦ/ТЦ — нет видимости, сложная навигация, офисный трафик.")
        else:
            barriers.append("🚨 КРИТИЧНО: Объект на 2+ этаже — нет видимости с улицы, сложная навигация.")
    if not params.get("separate_entrance", True):
        barriers.append("🚨 КРИТИЧНО: Нет отдельного входа — пациенту нужно заходить через общий подъезд/лестницу.")
    if not params.get("street_visibility", True):
        barriers.append("⚠️ Нет видимости с улицы — нет витрины/вывески, пациент не видит клинику.")
    if params.get("building_type") == "bc":
        barriers.append("⚠️ Бизнес-центр — офисный трафик, выходные пустые, парковка конкурирует с офисами.")
    if params.get("building_type") == "mall":
        barriers.append("⚠️ Торговый центр — сложная навигация, медицинский трафик нецелевой.")
    if not params.get("first_line", True):
        barriers.append("⚠️ Не первая линия — меньше проходного трафика, ниже узнаваемость.")

    if osm_available:
        if c.get("parking_500m", 0) == 0 and c.get("parking_1000m", 0) == 0:
            barriers.append("🚨 КРИТИЧНО: Нет парковки в радиусе 1 км по OSM.")
        elif c.get("parking_500m", 0) == 0:
            barriers.append("⚠️ Нет парковки в радиусе 500 м.")
        if full_profile.get("vehicle_access", 0) <= 25:
            barriers.append("🚨 КРИТИЧНО: Критически неудобный автомобильный подъезд.")
        if full_profile.get("public_transport", 0) <= 15:
            barriers.append("⚠️ Отсутствует общественный транспорт в пешей доступности.")
        if full_profile.get("competitor_density", 0) >= 85:
            barriers.append("⚠️ Очень высокая плотность конкурентов (3+ клиники в 300 м).")
        if full_profile.get("population_density", 0) <= 20:
            barriers.append("⚠️ Критически низкая плотность жилой застройки.")
    else:
        barriers.append("ℹ️ OSM-данные недоступны — барьеры по парковке, транспорту и застройке не проверены. Overpass API мог быть перегружен или заблокирован. Попробуйте перезапустить анализ.")

    return barriers


def apply_hard_penalties(absolute_score: float, full_profile: dict, barriers: List[str], params: dict, osm_available: bool) -> Tuple[float, float]:
    penalty = 0.0
    if params.get("floor") == "upper":
        penalty += 20
    if not params.get("separate_entrance", True):
        penalty += 15
    if not params.get("street_visibility", True):
        penalty += 10
    if params.get("building_type") == "bc":
        penalty += 12
    elif params.get("building_type") == "mall":
        penalty += 8
    if not params.get("first_line", True):
        penalty += 5

    if osm_available:
        if full_profile.get("parking_supply", 0) <= 10:
            penalty += 15
        elif full_profile.get("parking_supply", 0) <= 30:
            penalty += 8
        if full_profile.get("parking_proximity", 0) <= 15:
            penalty += 10
        elif full_profile.get("parking_proximity", 0) <= 35:
            penalty += 5
        if full_profile.get("vehicle_access", 0) <= 20:
            penalty += 12
        elif full_profile.get("vehicle_access", 0) <= 40:
            penalty += 5
        if full_profile.get("competitor_density", 0) >= 85:
            penalty += 8
        if full_profile.get("population_density", 0) <= 20:
            penalty += 10

    penalty = min(penalty, 50.0)
    final = round(clamp(absolute_score - penalty), 1)
    return final, penalty


# ==============================================================================
# CONFIDENCE
# ==============================================================================
def calculate_confidence(ai_profile: dict, osm: dict) -> int:
    ai_conf = float(ai_profile.get("profile_confidence", 50))
    evidence = float(ai_profile.get("evidence_quality", 50))
    osm_ok = 100 if osm.get("available") else 30
    raw = osm.get("raw_count", 0)
    if raw >= 80:
        osm_ok = min(100, osm_ok + 10)
    elif raw < 15:
        osm_ok = max(25, osm_ok - 20)
    return int(round(clamp(ai_conf * 0.40 + evidence * 0.35 + osm_ok * 0.25)))


# ==============================================================================
# FULL ANALYSIS
# ==============================================================================
def run_full_analysis(
    api_key: str,
    model: str,
    address: str,
    params: dict,
    known_data: dict,
    target_age: float,
    share_female: float,
    avg_ticket: int,
    сравнение_profiles: dict,
    status_callback=None,
) -> dict:
    target_lat, target_lon = get_exact_coordinates(address)
    coords_available = target_lat is not None and target_lon is not None
    if not coords_available:
        if status_callback:
            status_callback("1/3", "Координаты не определены — пропускаю OSM…")

    target_loc = {
        "address": address,
        "lat": target_lat if coords_available else 0.0,
        "lon": target_lon if coords_available else 0.0,
        "params": params,
        "target_age": target_age,
        "share_female": share_female,
        "avg_ticket": avg_ticket,
    }

    target_osm = {"available": False, "error": "no_coords", "counts": {}, "roads": {}, "landuse": {}, "buildings": {}, "raw_count": 0}
    osm_target_available = False
    if coords_available:
        if status_callback:
            status_callback("1/3", "Собираю OSM-данные для target (таймаут до 25 сек)…")
        target_osm = collect_osm_context(target_lat, target_lon)
        osm_target_available = target_osm.get("available", False)

    if status_callback:
        status_callback("2/3", "Запрашиваю AI-оценку target…")

    ai_failed = False
    try:
        full_ai = generate_target_profile(
            api_key=api_key,
            model=model.strip(),
            target_loc=target_loc,
            known_data=known_data,
            osm=target_osm,
        )
        target_profile = dict(full_ai)
    except Exception as exc:
        ai_failed = True
        target_profile = make_default_full_profile()

    loc_score, applied_penalties = compute_location_param_score(params)
    target_profile["location_param_score"] = loc_score

    target_ai = {k: target_profile.get(k, 50) for k in [
        "income_fit", "age_fit", "gender_fit", "family_profile", "daytime_balance",
        "competitor_strength", "market_gap", "noise_safety", "traffic_quality",
        "profile_confidence", "evidence_quality"
    ]}

    if osm_target_available:
        osm_scores = osm_to_factor_scores(target_osm)
        for k in osm_scores:
            target_profile[k] = osm_scores[k]

    # ПРИОРИТЕТ ПОЛЬЗОВАТЕЛЬСКИХ ДАННЫХ
    user_overrides = apply_user_overrides(target_profile, known_data)

    block_scores = compute_block_scores(target_profile)
    absolute_base = compute_absolute_score(block_scores)

    hard_barriers = calculate_hard_barriers(target_profile, target_osm, params)
    absolute_final, hard_penalty = apply_hard_penalties(absolute_base, target_profile, hard_barriers, params, osm_target_available)

    benchmark_rows = [
        {
            "address": v["address"],
            "status": v["status"],
            "profile": v["profile"],
        }
        for v in сравнение_profiles.values()
    ]

    сравнение = benchmark_analysis(target_profile, benchmark_rows)
    benchmark_valid = len(benchmark_rows) > 0

    if status_callback:
        status_callback("3/3", "Расчёт сравнение…")

    if benchmark_valid:
        benchmark_component = (
            сравнение["successful_centroid_similarity"] * 0.60 +
            clamp(50 + сравнение["benchmark_gap"] / 2) * 0.40
        )
    else:
        benchmark_component = 50.0

    final_score = round(absolute_final * 0.60 + benchmark_component * 0.40, 1)

    if final_score >= 75:
        verdict = "СИЛЬНАЯ ЛОКАЦИЯ"
    elif final_score >= 60:
        verdict = "ХОРОШАЯ ЛОКАЦИЯ С ОГОВОРКАМИ"
    elif final_score >= 45:
        verdict = "СРЕДНЯЯ ЛОКАЦИЯ"
    elif final_score >= 30:
        verdict = "СЛАБАЯ ЛОКАЦИЯ"
    else:
        verdict = "КРИТИЧЕСКИ СЛАБАЯ ЛОКАЦИЯ — НЕ РЕКОМЕНДУЕТСЯ"

    confidence = calculate_confidence(target_ai, target_osm)
    if confidence < 55:
        verdict += " — НИЗКАЯ УВЕРЕННОСТЬ"
    if not osm_target_available:
        verdict += " — ⚠️ OSM НЕДОСТУПЕН, ФАКТОРЫ ОЦЕНЕНЫ AI"
    if ai_failed:
        verdict += " — ⚠️ AI НЕДОСТУПЕН, НЕЙТРАЛЬНЫЕ ОЦЕНКИ"
    if not benchmark_valid:
        verdict += " — ⚠️ НЕТ РЕФЕРЕНСОВ, BENCHMARK НЕВОЗМОЖЕН"

    return {
        "address": address,
        "params": params,
        "applied_penalties": applied_penalties,
        "latitude": target_lat if coords_available else None,
        "longitude": target_lon if coords_available else None,
        "profile": target_profile,
        "block_scores": block_scores,
        "osm_context": target_osm,
        "osm_target_available": osm_target_available,
        "absolute_base": absolute_base,
        "absolute_score": absolute_final,
        "hard_penalty": hard_penalty,
        "hard_barriers": hard_barriers,
        "confidence": confidence,
        "benchmark": сравнение,
        "benchmark_valid": benchmark_valid,
        "benchmark_rows": benchmark_rows,
        "final_score": final_score,
        "verdict": verdict,
        "model": model,
        "ai_failed": ai_failed,
        "user_overrides": user_overrides,
    }


# ==============================================================================
# UI — API KEY
# ==============================================================================
if "openai_key" not in st.session_state:
    st.session_state.openai_key = None

if not st.session_state.openai_key:
    st.info("Введите OpenAI API-ключ. Хранится только в сессии.")
    key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    if st.button("Продолжить", type="primary"):
        if not key.strip():
            st.error("Введите ключ.")
        else:
            st.session_state.openai_key = key.strip()
            st.rerun()
    st.stop()


# ==============================================================================
# UI — ШАГ 1: ЗАГРУЗКА РЕФЕРЕНСОВ
# ==============================================================================
st.header("Шаг 1: Загрузите референсы")
st.caption("Загрузите Excel-файл. Ожидается Лист 2 со структурой как на скриншоте.")

uploaded = st.file_uploader("Файл с референсами (.xlsx)", type=["xlsx"])

df_benchmarks = None
if uploaded:
    try:
        if openpyxl is None:
            st.error("📦 Пакет `openpyxl` не установлен. Добавьте строку `openpyxl` в файл `requirements.txt` рядом с `app.py` и перезапустите приложение.")
            st.stop()
        df_benchmarks = pd.read_excel(uploaded, sheet_name=1)
        st.session_state.df_benchmarks = df_benchmarks
        st.success(f"Загружено {len(df_benchmarks)} референсов с Листа 2.")
        with st.expander("Предпросмотр загруженных данных"):
            st.dataframe(df_benchmarks, use_container_width=True)
    except Exception as exc:
        st.error(f"Ошибка чтения файла: {exc}")
        st.stop()
else:
    if "df_benchmarks" in st.session_state:
        df_benchmarks = st.session_state.df_benchmarks

if df_benchmarks is not None and "сравнение_profiles" not in st.session_state:
    if st.button("🤖 Сгенерировать профили референсов", type="primary"):
        with st.spinner("AI оценивает референсы (батч-запрос)…"):
            try:
                # Заменяем NaN на пустые строки для безопасности
                df_clean = df_benchmarks.fillna("")
                profiles = generate_benchmark_profiles(
                    api_key=st.session_state.openai_key,
                    model=DEFAULT_MODEL,
                    df=df_clean,
                )
                st.session_state.сравнение_profiles = profiles
                st.success(f"Профили сгенерированы для {len(profiles)} референсов.")
            except Exception as exc:
                st.error(f"Ошибка генерации профилей: {exc}")
                st.exception(exc)

if "сравнение_profiles" in st.session_state:
    st.info(f"✅ Референсы готовы: {len(st.session_state.сравнение_profiles)} шт.")


# ==============================================================================
# UI — ШАГ 2: ПАРАМЕТРЫ TARGET
# ==============================================================================
st.divider()
st.header("Шаг 2: Заполните известные данные о планируемом месте")

model = st.text_input("Модель OpenAI", value=DEFAULT_MODEL,
    help="Фиксированная модель. Если нет GPT-5.1 — укажите gpt-4o.")

st.subheader("🏢 Параметры объекта (критично)")
st.caption("Отметьте все, что соответствует вашему объекту.")

building_type = st.radio(
    "Тип здания",
    options=["residential", "standalone", "bc", "mall", "other"],
    format_func=lambda x: {
        "residential": "🏠 Жилой дом",
        "standalone": "🏢 Отдельное здание / особняк",
        "bc": "🏬 Бизнес-центр",
        "mall": "🛒 Торговый центр / пассаж",
        "other": "📦 Другое",
    }[x],
    horizontal=True,
)

col_f1, col_f2 = st.columns(2)
with col_f1:
    floor = st.radio(
        "Этаж",
        options=["ground", "upper"],
        format_func=lambda x: "1 этаж / цоколь" if x == "ground" else "2+ этаж",
        horizontal=True,
    )
with col_f2:
    first_line = st.toggle(
        "✅ Первая линия (главная улица)",
        value=True,
        help="Если объект во дворе или на второстепенной улице — отключите. Штраф −10.",
    )

col_c1, col_c2 = st.columns(2)
with col_c1:
    separate_entrance = st.toggle(
        "✅ Отдельный вход",
        value=True,
        help="Если вход через подъезд жилого дома, общий холл БЦ или лестницу — отключите. Штраф −20.",
    )
with col_c2:
    street_visibility = st.toggle(
        "✅ Видимость с улицы (витрина / вывеска)",
        value=True,
        help="Если нет витрины и вывеска не видна с улицы — отключите. Штраф −15.",
    )

params = {
    "building_type": building_type,
    "floor": floor,
    "separate_entrance": separate_entrance,
    "street_visibility": street_visibility,
    "first_line": first_line,
}
preview_score, preview_penalties = compute_location_param_score(params)

st.divider()
st.markdown("**Предварительная оценка параметров локации:**")
if preview_penalties:
    for name, penalty, desc in preview_penalties:
        st.warning(f"−{penalty} баллов: {name} — {desc}")
else:
    st.success("Все параметры оптимальны. Штрафов нет.")
st.info(f"**Базовый score параметров: {preview_score:.0f}/100** (максимум 100, минимум 5)")

st.subheader("👤 Портрет целевого пациента")
col1, col2, col3 = st.columns(3)
with col1:
    target_age = st.number_input("Средний возраст, лет", min_value=0, max_value=120, value=35, step=1)
with col2:
    share_female_percent = st.number_input("Доля женщин, %", min_value=0.0, max_value=100.0, value=60.0, step=1.0)
with col3:
    avg_ticket = st.number_input("Средний чек, руб.", min_value=0, max_value=1_000_000, value=3500, step=100)

st.subheader("📍 Адрес")
address = st.text_input("Адрес объекта", value="Екатеринбург, Энгельса, 36",
    placeholder="Например: Екатеринбург, Энгельса, 36")

st.subheader("📝 Известные данные о районе (опционально)")
st.caption("Если вы уже изучили локацию — заполните. Это повысит точность AI-оценки.")
known_data = {}

with st.expander("Заполнить известные данные"):
    kcol1, kcol2 = st.columns(2)
    with kcol1:
        known_data["parking"] = st.selectbox("Парковка", ["неизвестно", "да", "нет", "ограничена"], index=0)
        known_data["traffic_car"] = st.selectbox("Автотрафик", ["неизвестно", "экстремально_high", "высокий", "средний", "низкий"], index=0)
        known_data["traffic_ped"] = st.selectbox("Пеший трафик", ["неизвестно", "высокий", "средний", "низкий"], index=0)
        known_data["population_density"] = st.selectbox("Плотность населения", ["неизвестно", "очень_high", "высокая", "средняя", "ниже_medium", "низкая"], index=0)
        known_data["transport_access"] = st.selectbox("Транспортная доступность", ["неизвестно", "отличная", "очень_good", "хорошая", "средняя"], index=0)
    with kcol2:
        known_data["competitors_count"] = st.number_input("Количество конкурентов (если известно)", min_value=0, value=0, step=1)
        known_data["competitors_list"] = st.text_input("Список конкурентов", value="")
        known_data["price_segment"] = st.selectbox("Ценовой сегмент", ["неизвестно", "эконом", "средний", "средний+", "бизнес", "премиум"], index=0)
        known_data["has_trauma_center"] = st.selectbox("Травмпункт", ["неизвестно", "да", "нет"], index=0)
        known_data["building_type_1km"] = st.text_input("Тип застройки (1км)", value="")

known_data_clean = {k: v for k, v in known_data.items() if str(v).lower() not in ("неизвестно", "", "0", "nan")}


# ==============================================================================
# UI — ШАГ 3: АНАЛИЗ
# ==============================================================================
st.divider()
col_a, col_b = st.columns([2, 1])
with col_a:
    run_analysis = st.button("🔍 Запустить анализ", type="primary", use_container_width=True)
with col_b:
    clear_cache = st.button("♻️ Сбросить кэш", use_container_width=True)

if clear_cache:
    for k in ["last_result", "сравнение_profiles", "df_benchmarks"]:
        st.session_state.pop(k, None)
    st.rerun()

if run_analysis:
    if not address.strip():
        st.error("Адрес не должен быть пустым.")
        st.stop()
    if "сравнение_profiles" not in st.session_state:
        st.error("Сначала загрузите файл референсов и сгенерируйте профили (Шаг 1).")
        st.stop()

    share_female = share_female_percent / 100.0
    progress_box = st.empty()

    def update_status(step: str, text: str):
        progress_box.info(f"**{step}**  {text}")

    try:
        update_status("START", "Геокодирование и аудит…")
        result = run_full_analysis(
            api_key=st.session_state.openai_key,
            model=model.strip(),
            address=address.strip(),
            params=params,
            known_data=known_data_clean,
            target_age=float(target_age),
            share_female=float(share_female),
            avg_ticket=int(avg_ticket),
            сравнение_profiles=st.session_state.сравнение_profiles,
            status_callback=update_status,
        )
        st.session_state.last_result = result
        progress_box.success("✅ Анализ завершён.")
        st.rerun()
    except Exception as exc:
        progress_box.error("❌ Ошибка.")
        st.error(f"{type(exc).__name__}: {exc}")
        st.exception(exc)


# ==============================================================================
# OUTPUT
# ==============================================================================
if "last_result" in st.session_state:
    result = st.session_state.last_result
    if not isinstance(result, dict):
        st.error("Сохранённые данные повреждены. Сбросьте кэш и попробуйте снова.")
        st.stop()

    profile = result.get("profile", {})
    benchmark = result.get("benchmark", {})
    block_scores = result.get("block_scores", {})

    st.divider()
    st.subheader("📊 Результат анализа")
    st.markdown(f"### {result.get('address', '—')}")

    p = result.get("params", {})
    param_tags = []
    if p.get("building_type") == "bc": param_tags.append("🏬 БЦ")
    elif p.get("building_type") == "mall": param_tags.append("🛒 ТЦ")
    elif p.get("building_type") == "residential": param_tags.append("🏠 Жилой дом")
    elif p.get("building_type") == "standalone": param_tags.append("🏢 Отдельное здание")
    else: param_tags.append("📦 Другое")
    param_tags.append("2+ этаж" if p.get("floor") == "upper" else "1 этаж")
    if p.get("separate_entrance"): param_tags.append("✅ Отдельный вход")
    else: param_tags.append("❌ Нет отдельного входа")
    if p.get("street_visibility"): param_tags.append("✅ Видимость")
    else: param_tags.append("❌ Нет видимости")
    if p.get("first_line"): param_tags.append("✅ 1-я линия")
    else: param_tags.append("❌ Не 1-я линия")
    st.caption(" · ".join(param_tags))

    osm_target_available = result.get("osm_target_available", False)
    ai_failed = result.get("ai_failed", False)
    benchmark_valid = result.get("benchmark_valid", False)

    if not osm_target_available:
        st.warning("🤖 OSM недоступен (Overpass API не отвечает или заблокирован для облачных IP). Все факторы оценены AI на основе адреса и общих знаний. Попробуйте перезапустить анализ.")
    if ai_failed:
        st.error("🚨 OpenAI недоступен. Использованы нейтральные оценки 50. Результат может быть неточным.")
    if not benchmark_valid:
        st.error("🚨 Нет референсов — сравнение невозможен.")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Итоговая оценка", f"{result.get('final_score', 0)} / 100")
    with m2:
        st.metric("Абсолютное качество локации", f"{result.get('absolute_score', 0)} / 100")
    with m3:
        if benchmark_valid:
            st.metric("Схожесть с успешными референсами", f"{benchmark.get('successful_centroid_similarity', 0)} / 100")
        else:
            st.metric("Схожесть с успешными референсами", "N/A")
    with m4:
        st.metric("Уверенность", f"{result.get('confidence', 0)}%")

    final_score = result.get('final_score', 0)
    if final_score >= 75:
        st.success(f"### {result.get('verdict', '—')}")
    elif final_score >= 60:
        st.info(f"### {result.get('verdict', '—')}")
    elif final_score >= 45:
        st.warning(f"### {result.get('verdict', '—')}")
    else:
        st.error(f"### {result.get('verdict', '—')}")

    st.caption(f"Базовый оценка: {result.get('absolute_base', 0)}; hard-penalty: −{result.get('hard_penalty', 0)}")

    applied_penalties = result.get("applied_penalties", [])
    if applied_penalties:
        with st.expander("📐 Расчёт параметров локации"):
            st.markdown("База: **100** (идеальные параметры)")
            for name, penalty, desc in applied_penalties:
                st.markdown(f"−**{penalty}** — *{name}*: {desc}")
            st.markdown(f"**Итог: {profile.get('location_param_score', 0):.0f}/100**")

    # BENCHMARK
    if benchmark_valid:
        st.subheader("🎯 Сравнение")
        with st.expander("Как считается схожесть?"):
            st.markdown("""
**Формула:** взвешенное расстояние Манхэттена по всем факторам. Чем меньше разница между планируемой точкой и референсом — тем выше схожесть.

```
distance = Σ |target_i − сравнение_i| × weight_i  /  Σ weight_i
схожесть = 100 − distance
```

- 100% = профили идентичны (планируемая точка = референс)
- 0% = максимально разные
- Вес каждого фактора = вес_in_block × вес_block
""")

        bm1, bm2, bm3 = st.columns(3)
        success_sim = benchmark.get("success_similarity", [])
        weak_sim = benchmark.get("weak_similarity", [])
        with bm1:
            st.metric("Ближайший успешный референс", f"{success_sim[0][1]}%" if success_sim else "—")
        with bm2:
            st.metric("Средняя схожесть с успешными", f"{benchmark.get('successful_centroid_similarity', 0)}%")
        with bm3:
            st.metric("Средняя схожесть со слабыми", f"{benchmark.get('weak_centroid_similarity', 0)}%")

        st.metric("Разрыв с референсами", f"{benchmark.get('benchmark_gap', 0):+.1f}",
            help="Положительный = планируемая точка ближе к успешным референсам, чем к слабым. Отрицательный — наоборот.")

        # --- ВСЕ РЕФЕРЕНСЫ (ranked) ---
        st.subheader("📊 Сравнение со всеми референсами")
        all_sim = benchmark.get("all_similarity", [])
        if all_sim:
            all_data = []
            for rank, (addr, sim, status) in enumerate(all_sim, 1):
                all_data.append({
                    "#": rank,
                    "Референс": addr,
                    "Схожесть": sim,
                    "Группа": "🟢 Успешный" if status in ("успешная", "успешный") else "🔴 Слабый",
                })
            df_all = pd.DataFrame(all_data)
            st.dataframe(df_all, use_container_width=True, hide_index=True, height=320)

        # --- Детальный разбор ---
        st.subheader("🔍 Детальный разбор по референсам")
        st.caption("Для каждого референса показано: насколько планируемая точка (слева) отличается от референса (справа). Весовой вклад — насколько этот фактор влияет на итоговую схожесть.")
        all_debug = benchmark.get("all_debug", [])
        for addr, sim, debug, status in all_debug:
            label = f"{addr} — {sim:.1f}% ({status})"
            with st.expander(label):
                st.markdown("**Топ-10 факторов с наибольшим весовым вкладом:**")
                for factor, t_val, b_val, contrib in debug[:10]:
                    delta = t_val - b_val
                    arrow = "🟢" if abs(delta) < 5 else ("🟡" if abs(delta) < 15 else "🔴")
                    st.markdown(f"{arrow} **{factor}**: планируемая точка={t_val}, референс={b_val}, разница={delta:+.1f}, весовой вклад={contrib}")
                st.markdown("---")
                st.markdown("**Полный разбор:**")
                full_df = pd.DataFrame([
                    {"Фактор": f, "Планируемая точка": t, "Сравнение с референсами": b, "Разница": round(t-b,1), "Весовой вклад": c}
                    for f, t, b, c in debug
                ])
                st.dataframe(full_df, use_container_width=True, hide_index=True, height=560)

    # BLOCKS
    st.subheader("🧭 Сводка по блокам")
    st.caption("Оценка по 6 группам факторов. Каждый блок взвешен — сумма весов = 100%.")
    block_labels = {
        "location_params": "Параметры локации",
        "parking_access": "Парковка и доступность",
        "demand": "Спрос и ЦА",
        "competition": "Конкуренция",
        "medical_eco": "Гос. медицина и кластер",
        "visibility_env": "Видимость и среда",
    }
    block_df = pd.DataFrame([
        {"Блок": block_labels.get(b, b), "Оценка": block_scores.get(b, 0), "Вес": f"{BLOCK_WEIGHTS.get(b, 0)*100:.0f}%"}
        for b in BLOCK_WEIGHTS
    ])
    st.dataframe(block_df, use_container_width=True, hide_index=True)

    # USER OVERRIDES
    user_overrides = result.get("user_overrides", {})
    if user_overrides:
        st.subheader("👤 Переопределения пользователем")
        st.caption("Вы ввели конкретные данные о районе — они имеют приоритет над AI и OSM.")
        for factor, (old_val, new_val, reason) in user_overrides.items():
            delta = new_val - old_val
            if delta < -15:
                icon = "🔴"
                color = "error"
            elif delta < 0:
                icon = "🟠"
                color = "warning"
            elif delta > 15:
                icon = "🟢"
                color = "success"
            else:
                icon = "🟡"
                color = "info"
            st.markdown(f"{icon} **{FACTOR_LABEL.get(factor, factor)}**: {old_val:.0f} → **{new_val:.0f}** ({delta:+.0f}) — *{reason}*")

    # HARD BARRIERS
    st.subheader("🚨 Жёсткие барьеры и риски")
    st.caption("Барьеры, которые гарантированно снижают оценку. 🚨 Критично = минус 15–20 баллов. ⚠️ Предупреждение = минус 5–10 баллов.")
    hard_barriers = result.get("hard_barriers", [])
    if hard_barriers:
        for b in hard_barriers:
            if "КРИТИЧНО" in b:
                st.error(b)
            else:
                st.warning(b)
    else:
        st.success("Критических барьеров не выявлено.")

    # FACTORS DETAIL
    st.subheader("🔎 Детализация факторов")
    st.caption("🟢 Отлично (75+) · 🟡 Нормально (50–74) · 🟠 Плохо (30–49) · 🔴 Критично (<30)")
    rows = []
    for factor in FACTOR_KEYS:
        block = FACTOR_BLOCK.get(factor, "")
        value = profile.get(factor, 0)
        if factor in LOW_IS_BAD:
            suitability = 100.0 - value
        else:
            suitability = value
        src = FACTOR_SOURCE.get(factor, "")
        src_icon = {"osm": "🗺️ OSM", "ai": "🤖 AI", "user": "👤 Пользователь", "rule": "📐 Правило"}.get(src, "")
        if suitability >= 75:
            status = "🟢"
        elif suitability >= 50:
            status = "🟡"
        elif suitability >= 30:
            status = "🟠"
        else:
            status = "🔴"
        desc = FACTOR_DESCRIPTIONS.get(factor, "")
        rows.append({
            "": status,
            "Блок": block_labels.get(block, block),
            "Фактор": FACTOR_LABEL.get(factor, factor),
            "Оценка": round(suitability, 1),
            "Источник": src_icon,
            "Описание": desc,
        })
    df_f = pd.DataFrame(rows)
    st.dataframe(df_f, use_container_width=True, hide_index=True, height=700)

    # STRENGTHS / RISKS
    st.subheader("💪 Сильные стороны")
    strong = df_f[df_f["Оценка"] >= 75].head(10)
    if strong.empty:
        st.write("Нет факторов ≥75.")
    else:
        for _, row in strong.iterrows():
            st.markdown(f"🟢 **{row['Фактор']}** — {row['Оценка']:.0f}/100 ({row['Источник']})")
            st.caption(row['Описание'])

    st.subheader("⚠️ Ограничения")
    weak_factors = df_f[df_f["Оценка"] < 50].sort_values("Оценка").head(12)
    if weak_factors.empty:
        st.success("Нет факторов ниже 50/100.")
    else:
        for _, row in weak_factors.iterrows():
            st.markdown(f"🔴 **{row['Фактор']}** — {row['Оценка']:.0f}/100 ({row['Источник']})")
            st.caption(row['Описание'])

    # OSM AUDIT
    st.subheader("🗺️ OSM-аудит")
    st.caption("Данные OpenStreetMap — объективные данные о парковках, дорогах, застройке и медучреждениях вокруг планируемой точки.")
    osm = result.get("osm_context", {})
    if osm.get("available"):
        st.success(f"OSM доступен. Элементов: {osm.get('raw_count', 0)}.")
        osm_counts = osm.get("counts", {})
        osm_df = pd.DataFrame([{"Показатель": k, "Количество": v} for k, v in osm_counts.items()])
        st.dataframe(osm_df, use_container_width=True, hide_index=True)
    else:
        st.info(f"🤖 OSM недоступен ({osm.get('error', 'unknown')}). Все факторы оценены AI.")

    lat_disp = result.get('latitude')
    lon_disp = result.get('longitude')
    coord_str = f"{lat_disp:.6f}, {lon_disp:.6f}" if lat_disp is not None and lon_disp is not None else "не определены"
    st.caption(f"Координаты: {coord_str} · Модель: {result.get('model', model)}")
    with st.expander("Показать полный профиль (JSON)"):
        st.json(profile)


# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.header("Сессия")
    st.success("OpenAI API-ключ активен.")
    st.markdown("""
### Архитектура v4.0

**Шаг 1 — Референсы (эталонные клиники)**
- Загрузка Excel (Лист 2) с существующими клиниками
- AI оценивает все факторы для каждого референса батчем
- Референсы делятся на успешные и слабые — для сравнения

**Шаг 2 — Планируемая точка (новая клиника)**
- Параметры объекта (этаж, вход, видимость и т.д.)
- Опциональные данные о районе (повышают точность)
- Адрес + портрет целевого пациента

**Шаг 3 — Анализ**
- Геокодирование target
- OSM (1 запрос, 5 сек)
- AI-оценка target (все 21 фактор)
- Сравнение с референсами

**Hard rules:** штрафы до 50 баллов
- OSM-зависимые только при доступном OSM

**Сравнение:** планируемая точка vs референсы
- Статус референса: успешная / слабая / спорная
- Чем выше схожесть с успешными — тем лучше локация
""")
    if st.button("Сбросить OpenAI ключ"):
        st.session_state.clear()
        st.rerun()
    st.caption("Используйте одну модель и не меняйте референсы без пересчёта.")
