# -*- coding: utf-8 -*-
"""
GeoMarketing AI — Clinic Location Benchmark v3.6

Критические исправления:
1. OSM только для target (1 запрос, таймаут 5 сек). Benchmark — статические профили.
2. Если OSM недоступен — AI оценивает ВСЕ 21 фактор (не нейтральные 50).
3. Hard barriers/penalties: OSM-зависимые только при реальных данных.
4. Пустой ответ от Overpass = unavailable (не нули).
5. UI защита от KeyError.
"""

import json
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from openai import OpenAI
from pydantic import BaseModel, Field

# ==============================================================================
# STREAMLIT CONFIG
# ==============================================================================
st.set_page_config(
    page_title="Геомаркетинг клиники — Benchmark v3.6",
    page_icon="📍",
    layout="wide",
)

st.title("📍 Геомаркетинговый анализ локации клиники — v3.6")
st.caption("Явные параметры + детерминированный скоринг. Платные geo-API не нужны.")

# ==============================================================================
# КОНФИГУРАЦИЯ
# ==============================================================================
DEFAULT_MODEL = "gpt-5.1"
MODEL_REASONING = "low"

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

REQUEST_HEADERS = {
    "User-Agent": "ClinicGeoAnalytics/3.6 (streamlit-cloud; business use)"
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
    ("market_gap",              "competition",     0.25, "ai",    "Рыночный зазор"),
    ("pharmacy_synergy",        "medical_eco",     0.25, "osm",   "Синергия с аптеками"),
    ("diagnostics_synergy",     "medical_eco",     0.25, "osm",   "Синергия с диагностикой"),
    ("hospital_synergy",        "medical_eco",     0.25, "osm",   "Синергия с больницами"),
    ("medical_cluster",         "medical_eco",     0.25, "osm",   "Медицинский кластер"),
    ("visibility",              "visibility_env",  0.35, "osm",   "Видимость с дороги"),
    ("road_type_fit",           "visibility_env",  0.25, "osm",   "Тип дорог (жилой/офисный)"),
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

# ==============================================================================
# ЭТАЛОНЫ (benchmark) — предвычисленные профили. OSM не нужен.
# ==============================================================================
BENCHMARK_PROFILES = {
    "benchmark_1": {  # Красноярск, ул. 9 Мая, 19а — успешный
        "location_param_score": 100,
        "parking_proximity": 70, "parking_supply": 65, "vehicle_access": 75,
        "public_transport": 70, "population_density": 80,
        "income_fit": 65, "age_fit": 70, "gender_fit": 65, "family_profile": 75,
        "daytime_balance": 65,
        "competitor_density": 35, "competitor_strength": 65, "market_gap": 60,
        "pharmacy_synergy": 75, "diagnostics_synergy": 60, "hospital_synergy": 50,
        "medical_cluster": 70,
        "visibility": 80, "road_type_fit": 75, "pedestrian_comfort": 70,
        "noise_safety": 65, "traffic_quality": 70,
    },
    "benchmark_2": {  # Красноярск, ул. Ладо Кецховели, 34 — успешный
        "location_param_score": 100,
        "parking_proximity": 65, "parking_supply": 60, "vehicle_access": 70,
        "public_transport": 65, "population_density": 75,
        "income_fit": 60, "age_fit": 65, "gender_fit": 60, "family_profile": 70,
        "daytime_balance": 60,
        "competitor_density": 30, "competitor_strength": 70, "market_gap": 65,
        "pharmacy_synergy": 70, "diagnostics_synergy": 55, "hospital_synergy": 45,
        "medical_cluster": 65,
        "visibility": 75, "road_type_fit": 70, "pedestrian_comfort": 65,
        "noise_safety": 60, "traffic_quality": 65,
    },
    "benchmark_3": {  # Екатеринбург, ул. Советская, 42 — успешный
        "location_param_score": 100,
        "parking_proximity": 80, "parking_supply": 75, "vehicle_access": 85,
        "public_transport": 85, "population_density": 70,
        "income_fit": 75, "age_fit": 70, "gender_fit": 70, "family_profile": 65,
        "daytime_balance": 80,
        "competitor_density": 50, "competitor_strength": 55, "market_gap": 55,
        "pharmacy_synergy": 85, "diagnostics_synergy": 75, "hospital_synergy": 70,
        "medical_cluster": 80,
        "visibility": 90, "road_type_fit": 85, "pedestrian_comfort": 75,
        "noise_safety": 55, "traffic_quality": 80,
    },
    "benchmark_4": {  # Казань, ул. Алексея Козина, 2 — успешный
        "location_param_score": 100,
        "parking_proximity": 70, "parking_supply": 65, "vehicle_access": 75,
        "public_transport": 75, "population_density": 80,
        "income_fit": 70, "age_fit": 75, "gender_fit": 75, "family_profile": 80,
        "daytime_balance": 70,
        "competitor_density": 40, "competitor_strength": 60, "market_gap": 60,
        "pharmacy_synergy": 75, "diagnostics_synergy": 65, "hospital_synergy": 55,
        "medical_cluster": 70,
        "visibility": 80, "road_type_fit": 75, "pedestrian_comfort": 75,
        "noise_safety": 70, "traffic_quality": 75,
    },
    "benchmark_5": {  # Новосибирск, ул. Новогодняя, 23/1 — слабый (БЦ, 2 этаж)
        "location_param_score": 25,
        "parking_proximity": 40, "parking_supply": 35, "vehicle_access": 50,
        "public_transport": 45, "population_density": 40,
        "income_fit": 55, "age_fit": 50, "gender_fit": 50, "family_profile": 40,
        "daytime_balance": 85,
        "competitor_density": 70, "competitor_strength": 40, "market_gap": 30,
        "pharmacy_synergy": 40, "diagnostics_synergy": 35, "hospital_synergy": 30,
        "medical_cluster": 35,
        "visibility": 25, "road_type_fit": 40, "pedestrian_comfort": 35,
        "noise_safety": 50, "traffic_quality": 40,
    },
    "benchmark_6": {  # Челябинск, ул. Худякова, 10 — слабый
        "location_param_score": 75,
        "parking_proximity": 50, "parking_supply": 45, "vehicle_access": 60,
        "public_transport": 50, "population_density": 55,
        "income_fit": 50, "age_fit": 50, "gender_fit": 50, "family_profile": 55,
        "daytime_balance": 55,
        "competitor_density": 55, "competitor_strength": 45, "market_gap": 40,
        "pharmacy_synergy": 50, "diagnostics_synergy": 40, "hospital_synergy": 35,
        "medical_cluster": 45,
        "visibility": 30, "road_type_fit": 55, "pedestrian_comfort": 50,
        "noise_safety": 55, "traffic_quality": 50,
    },
    "benchmark_7": {  # Самара, ул. Академика Платонова, 10 корпус 3 — слабый
        "location_param_score": 75,
        "parking_proximity": 45, "parking_supply": 40, "vehicle_access": 55,
        "public_transport": 45, "population_density": 60,
        "income_fit": 50, "age_fit": 50, "gender_fit": 50, "family_profile": 60,
        "daytime_balance": 55,
        "competitor_density": 50, "competitor_strength": 50, "market_gap": 45,
        "pharmacy_synergy": 45, "diagnostics_synergy": 40, "hospital_synergy": 35,
        "medical_cluster": 40,
        "visibility": 30, "road_type_fit": 50, "pedestrian_comfort": 55,
        "noise_safety": 55, "traffic_quality": 50,
    },
}

DATA_CLINICS = [
    {"address": "Красноярск, ул. 9 Мая, 19а",       "status": "успешный", "lat": 56.067749, "lon": 92.933822, "params": {"building_type": "residential", "floor": "ground", "separate_entrance": True, "street_visibility": True, "first_line": True}, "key": "benchmark_1"},
    {"address": "Красноярск, ул. Ладо Кецховели, 34", "status": "успешный", "lat": 56.017160, "lon": 92.813882, "params": {"building_type": "residential", "floor": "ground", "separate_entrance": True, "street_visibility": True, "first_line": True}, "key": "benchmark_2"},
    {"address": "Екатеринбург, ул. Советская, 42",   "status": "успешный", "lat": 56.855058, "lon": 60.639260, "params": {"building_type": "standalone", "floor": "ground", "separate_entrance": True, "street_visibility": True, "first_line": True}, "key": "benchmark_3"},
    {"address": "Казань, ул. Алексея Козина, 2",     "status": "успешный", "lat": 55.814523, "lon": 49.141033, "params": {"building_type": "residential", "floor": "ground", "separate_entrance": True, "street_visibility": True, "first_line": True}, "key": "benchmark_4"},
    {"address": "Новосибирск, ул. Новогодняя, 23/1", "status": "слабый",   "lat": 54.987320, "lon": 82.911925, "params": {"building_type": "bc", "floor": "upper", "separate_entrance": False, "street_visibility": False, "first_line": False}, "key": "benchmark_5"},
    {"address": "Челябинск, ул. Худякова, 10",       "status": "слабый",   "lat": 55.148154, "lon": 61.365313, "params": {"building_type": "other", "floor": "ground", "separate_entrance": True, "street_visibility": False, "first_line": False}, "key": "benchmark_6"},
    {"address": "Самара, ул. Академика Платонова, 10 корпус 3", "status": "слабый", "lat": 53.218579, "lon": 50.176465, "params": {"building_type": "residential", "floor": "ground", "separate_entrance": True, "street_visibility": False, "first_line": False}, "key": "benchmark_7"},
]


# ==============================================================================
# PYDANTIC
# ==============================================================================
class GeoAIProfile(BaseModel):
    """9 AI-факторов (когда OSM доступен)."""
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


class GeoAIFullProfile(BaseModel):
    """Все 21 фактор (когда OSM недоступен — AI оценивает всё)."""
    # OSM-факторы
    parking_proximity: int = Field(ge=0, le=100)
    parking_supply: int = Field(ge=0, le=100)
    vehicle_access: int = Field(ge=0, le=100)
    public_transport: int = Field(ge=0, le=100)
    population_density: int = Field(ge=0, le=100)
    competitor_density: int = Field(ge=0, le=100)
    pharmacy_synergy: int = Field(ge=0, le=100)
    diagnostics_synergy: int = Field(ge=0, le=100)
    hospital_synergy: int = Field(ge=0, le=100)
    medical_cluster: int = Field(ge=0, le=100)
    visibility: int = Field(ge=0, le=100)
    road_type_fit: int = Field(ge=0, le=100)
    pedestrian_comfort: int = Field(ge=0, le=100)
    # AI-факторы
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


class GeoProfileItem(BaseModel):
    key: str
    profile: GeoAIProfile


class GeoProfileBatch(BaseModel):
    profiles: List[GeoProfileItem]


class GeoProfileItemFull(BaseModel):
    key: str
    profile: GeoAIFullProfile


class GeoProfileBatchFull(BaseModel):
    profiles: List[GeoProfileItemFull]


# ==============================================================================
# OPENAI
# ==============================================================================
def call_batch_ai(client: OpenAI, model: str, system_prompt: str, user_prompt: str) -> Optional[GeoProfileBatch]:
    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=GeoProfileBatch,
        timeout=120,
    )
    if model.startswith("gpt-5"):
        kwargs["reasoning_effort"] = MODEL_REASONING
    response = client.beta.chat.completions.parse(**kwargs)
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise ValueError("OpenAI не вернул структурированный ответ.")
    return parsed


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


def make_default_ai_profile() -> dict:
    return {
        "income_fit": 50, "age_fit": 50, "gender_fit": 50, "family_profile": 50,
        "daytime_balance": 50, "competitor_strength": 50, "market_gap": 50,
        "noise_safety": 50, "traffic_quality": 50, "profile_confidence": 30, "evidence_quality": 25,
    }


def make_default_full_profile() -> dict:
    return {
        "parking_proximity": 50, "parking_supply": 50, "vehicle_access": 50,
        "public_transport": 50, "population_density": 50, "competitor_density": 50,
        "pharmacy_synergy": 50, "diagnostics_synergy": 50, "hospital_synergy": 50,
        "medical_cluster": 50, "visibility": 50, "road_type_fit": 50, "pedestrian_comfort": 50,
        "income_fit": 50, "age_fit": 50, "gender_fit": 50, "family_profile": 50,
        "daytime_balance": 50, "competitor_strength": 50, "market_gap": 50,
        "noise_safety": 50, "traffic_quality": 50, "profile_confidence": 30, "evidence_quality": 25,
    }


# ==============================================================================
# ГЕОКОДИРОВАНИЕ
# ==============================================================================
@st.cache_data(show_spinner=False, ttl=86400)
def get_exact_coordinates(address: str) -> Tuple[Optional[float], Optional[float]]:
    for attempt in range(3):
        try:
            response = requests.get(
                NOMINATIM_URL,
                params={"q": address, "format": "jsonv2", "limit": 1, "addressdetails": 1},
                headers=REQUEST_HEADERS,
                timeout=12,
            )
            response.raise_for_status()
            data = response.json()
            if not data:
                if attempt < 2:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                return None, None
            return float(data[0]["lat"]), float(data[0]["lon"])
        except Exception:
            if attempt < 2:
                time.sleep(1.0 * (attempt + 1))
            continue
    return None, None


# ==============================================================================
# OVERPASS / OSM  (только target, таймаут 5 сек, пустой ответ = unavailable)
# ==============================================================================
def _overpass_request(query: str) -> List[dict]:
    try:
        response = requests.post(OVERPASS_URL, data={"data": query}, headers=REQUEST_HEADERS, timeout=5)
        response.raise_for_status()
        return response.json().get("elements", [])
    except Exception:
        return []


def collect_osm_context(lat: float, lon: float) -> dict:
    """Один лёгкий запрос для target. Без @st.cache_data — всегда свежий."""
    query = f"""
    [out:json][timeout:5];
    (
      nwr(around:800,{lat},{lon})["amenity"~"pharmacy|hospital|clinic|doctors"];
      nwr(around:800,{lat},{lon})["healthcare"~"centre|clinic|doctor|laboratory|diagnostic"];
      nwr(around:1000,{lat},{lon})["amenity"="parking"];
      nwr(around:300,{lat},{lon})["highway"~"bus_stop|platform"];
      nwr(around:800,{lat},{lon})["public_transport"];
      nwr(around:1000,{lat},{lon})["building"~"apartments|residential|house|detached|office|commercial|retail"];
      nwr(around:800,{lat},{lon})["highway"~"primary|secondary|tertiary|residential"];
    );
    out center tags;
    """
    try:
        elements = _overpass_request(query)
    except Exception as exc:
        return {"available": False, "error": str(exc), "counts": {}, "roads": {}, "landuse": {}, "buildings": {}, "raw_count": 0}

    if not elements:
        return {"available": False, "error": "empty_or_timeout", "counts": {}, "roads": {}, "landuse": {}, "buildings": {}, "raw_count": 0}

    counts = {
        "pharmacy_300m": 0, "pharmacy_800m": 0,
        "clinic_300m": 0, "clinic_800m": 0,
        "hospital_300m": 0, "hospital_800m": 0,
        "diag_lab_300m": 0, "diag_lab_800m": 0,
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

        if amenity == "pharmacy":
            counts["pharmacy_800m"] += 1
        if amenity in ("clinic", "doctors") or healthcare in ("clinic", "doctor", "centre"):
            counts["clinic_800m"] += 1
        if amenity == "hospital" or healthcare == "hospital":
            counts["hospital_800m"] += 1
        if healthcare in ("laboratory", "diagnostic"):
            counts["diag_lab_800m"] += 1
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

    # Прокси для _300m/_500m из _800m/_1000m
    counts["pharmacy_300m"] = counts["pharmacy_800m"]
    counts["clinic_300m"] = counts["clinic_800m"]
    counts["hospital_300m"] = counts["hospital_800m"]
    counts["diag_lab_300m"] = counts["diag_lab_800m"]
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

    clinic_300 = c.get("clinic_300m", 0)
    clinic_800 = c.get("clinic_800m", 0)
    if clinic_300 >= 3 or clinic_800 >= 8:
        scores["competitor_density"] = 95
    elif clinic_300 >= 2 or clinic_800 >= 5:
        scores["competitor_density"] = 75
    elif clinic_300 >= 1 or clinic_800 >= 3:
        scores["competitor_density"] = 50
    elif clinic_800 >= 1:
        scores["competitor_density"] = 25
    else:
        scores["competitor_density"] = 5

    ph_300 = c.get("pharmacy_300m", 0)
    ph_800 = c.get("pharmacy_800m", 0)
    if ph_300 >= 2:
        scores["pharmacy_synergy"] = 95
    elif ph_300 >= 1 or ph_800 >= 3:
        scores["pharmacy_synergy"] = 75
    elif ph_800 >= 1:
        scores["pharmacy_synergy"] = 50
    else:
        scores["pharmacy_synergy"] = 25

    diag_300 = c.get("diag_lab_300m", 0)
    diag_800 = c.get("diag_lab_800m", 0)
    if diag_300 >= 1:
        scores["diagnostics_synergy"] = 90
    elif diag_800 >= 2:
        scores["diagnostics_synergy"] = 70
    elif diag_800 >= 1:
        scores["diagnostics_synergy"] = 45
    else:
        scores["diagnostics_synergy"] = 20

    hosp_300 = c.get("hospital_300m", 0)
    hosp_800 = c.get("hospital_800m", 0)
    if hosp_300 >= 1:
        scores["hospital_synergy"] = 90
    elif hosp_800 >= 1:
        scores["hospital_synergy"] = 65
    else:
        scores["hospital_synergy"] = 30

    med_total = ph_800 + c.get("clinic_800m", 0) + hosp_800 + diag_800
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


# ==============================================================================
# AI PROMPT
# ==============================================================================
def build_ai_system_prompt() -> str:
    return """Ты — geo-marketing analyst для частных медицинских клиник формата «клиника у дома».

Твоя задача — оценить ТОЛЬКО следующие факторы по шкале 0–100:
1. income_fit — соответствие дохода населения среднему чеку клиники.
2. age_fit — возрастное соответствие целевой аудитории.
3. gender_fit — половое соответствие ЦА (если клиника женская — важно).
4. family_profile — семейный профиль района (дети, семьи).
5. daytime_balance — баланс дневного и жилого населения.
6. competitor_strength — сила конкурентов (100 = слабые/отсутствуют, 0 = сильные сетевые).
7. market_gap — рыночный зазор (100 = большой незакрытый спрос).
8. noise_safety — шумовая обстановка и безопасность (100 = тихо и безопасно).
9. traffic_quality — качество трафика для ЦА (не количество, а соответствие ЦА).

ПРАВИЛА:
- Используй только переданный адрес, город и OSM-контекст.
- Не оценивай парковку, видимость, доступность — это считается отдельно.
- Не используй статус успешный/слабый (он не передан).
- Делай экспертные оценки. Если данных мало — снижай confidence.
- Все оценки целые числа 0–100.
"""


def build_ai_full_system_prompt() -> str:
    return """Ты — geo-marketing analyst для частных медицинских клиник формата «клиника у дома».

OSM-данные для этой локации недоступны. Твоя задача — оценить ВСЕ факторы самостоятельно на основе адреса, города и своих знаний о районе.

Оцени по шкале 0–100:
1. parking_proximity — близость парковки (0 = нет парковки в радиусе 1 км, 100 = много парковки прямо у входа).
2. parking_supply — ёмкость парковки (0 = нет мест, 100 = огромная многоуровневая парковка).
3. vehicle_access — удобство подъезда на авто (0 = глухой переулок, 100 = широкая магистраль с удобным съездом).
4. public_transport — общественный транспорт (0 = нет остановок в пешей доступности, 100 = метро + несколько маршрутов).
5. population_density — плотность жилой застройки (0 = промзона/пустырь, 100 = плотная многоэтажная застройка).
6. competitor_density — плотность конкурентов (0 = нет клиник в радиусе 1 км, 100 = 3+ клиники в 300 м). ВНИМАНИЕ: 100 = много конкурентов (это ПЛОХО).
7. pharmacy_synergy — синергия с аптеками (0 = нет аптек, 100 = 2+ аптеки в 300 м).
8. diagnostics_synergy — синергия с диагностикой (0 = нет лабораторий, 100 = диагностический центр рядом).
9. hospital_synergy — синергия с больницами (0 = нет больниц, 100 = крупная больница в 300 м).
10. medical_cluster — медицинский кластер (0 = нет медучреждений, 100 = медицинский квартал).
11. visibility — видимость с дороги (0 = двор/подвал, 100 = витрина на главной магистрали).
12. road_type_fit — тип дорог (0 = промзона/трасса, 100 = жилой район с хорошим трафиком ЦА).
13. pedestrian_comfort — пешеходный комфорт (0 = нет тротуаров, 100 = широкие тротуары, озеленение).
14. income_fit — соответствие дохода населения среднему чеку клиники.
15. age_fit — возрастное соответствие целевой аудитории.
16. gender_fit — половое соответствие ЦА.
17. family_profile — семейный профиль района.
18. daytime_balance — баланс дневного и жилого населения.
19. competitor_strength — сила конкурентов (100 = слабые/отсутствуют).
20. market_gap — рыночный зазор (100 = большой незакрытый спрос).
21. noise_safety — шум и безопасность (100 = тихо и безопасно).
22. traffic_quality — качество трафика для ЦА.

ПРАВИЛА:
- Используй свои знания о городе и районе по адресу.
- Если района не знаешь — делай разумные предположения по типу улицы и города.
- Все оценки целые числа 0–100.
- Снижай confidence и evidence_quality если мало данных.
"""


def build_batch_user_prompt(locations: List[dict], osm_by_key: Dict[str, dict], target_key: str) -> str:
    chunks = []
    for loc in locations:
        key = loc["key"]
        osm = osm_by_key.get(key, {})
        p = loc.get("params", {})
        param_desc = []
        if p.get("building_type") == "bc": param_desc.append("Бизнес-центр")
        elif p.get("building_type") == "mall": param_desc.append("Торговый центр")
        elif p.get("building_type") == "residential": param_desc.append("Жилой дом")
        elif p.get("building_type") == "standalone": param_desc.append("Отдельное здание")
        else: param_desc.append("Другое")
        param_desc.append("2+ этаж" if p.get("floor") == "upper" else "1 этаж")
        param_desc.append("Отдельный вход" if p.get("separate_entrance") else "Нет отдельного входа")
        param_desc.append("Видимость с улицы" if p.get("street_visibility") else "Нет видимости")
        param_desc.append("Первая линия" if p.get("first_line") else "Не первая линия")
        osm_status = "✅ Доступен" if osm.get("available") else "❌ Недоступен"
        chunks.append(f"""
--- ЛОКАЦИЯ {key} ---
Адрес: {loc["address"]}
Параметры: {', '.join(param_desc)}
Координаты: {loc["lat"]:.6f}, {loc["lon"]:.6f}
ЦА: возраст {loc["target_age"]:.0f}; женщины {loc["share_female"]*100:.0f}%; чек {loc["avg_ticket"]:,} руб.
OSM статус: {osm_status}
OSM counts: {json.dumps(osm.get("counts", {}), ensure_ascii=False)}
""")
    return f"""Построй профиль для КАЖДОЙ локации из списка ниже.
Target-локация: {target_key}

КРИТИЧНО:
- Верни РОВНО по одному профилю на каждый key.
- Оценивай независимо, НЕ сравнивай локации между собой.
- Не используй статус successful/weak (не передан).

{''.join(chunks)}
"""


@st.cache_data(show_spinner=False, ttl=3600)
def generate_profiles_batch_cached(api_key: str, model: str, locations_json: str, osm_json: str) -> dict:
    client = OpenAI(api_key=api_key)
    locations = json.loads(locations_json)
    osm_by_key = json.loads(osm_json)
    batch = call_batch_ai(
        client=client,
        model=model,
        system_prompt=build_ai_system_prompt(),
        user_prompt=build_batch_user_prompt(locations, osm_by_key, locations[0]["key"]),
    )
    if batch is None:
        raise ValueError("OpenAI вернул None.")
    result = {item.key: item.profile.model_dump() for item in batch.profiles}
    expected = {loc["key"] for loc in locations}
    missing = expected - set(result)
    if missing:
        raise ValueError(f"OpenAI не вернул профили для: {', '.join(sorted(missing))}")
    return result


@st.cache_data(show_spinner=False, ttl=3600)
def generate_full_profile_cached(api_key: str, model: str, location_json: str) -> dict:
    """AI оценивает ВСЕ 21 фактор для одной локации (когда OSM недоступен)."""
    client = OpenAI(api_key=api_key)
    loc = json.loads(location_json)
    p = loc.get("params", {})
    param_desc = []
    if p.get("building_type") == "bc": param_desc.append("Бизнес-центр")
    elif p.get("building_type") == "mall": param_desc.append("Торговый центр")
    elif p.get("building_type") == "residential": param_desc.append("Жилой дом")
    elif p.get("building_type") == "standalone": param_desc.append("Отдельное здание")
    else: param_desc.append("Другое")
    param_desc.append("2+ этаж" if p.get("floor") == "upper" else "1 этаж")
    param_desc.append("Отдельный вход" if p.get("separate_entrance") else "Нет отдельного входа")
    param_desc.append("Видимость с улицы" if p.get("street_visibility") else "Нет видимости")
    param_desc.append("Первая линия" if p.get("first_line") else "Не первая линия")

    user_prompt = f"""Оцени ВСЕ факторы для локации:

Адрес: {loc["address"]}
Параметры: {', '.join(param_desc)}
Координаты: {loc["lat"]:.6f}, {loc["lon"]:.6f}
ЦА: возраст {loc["target_age"]:.0f}; женщины {loc["share_female"]*100:.0f}%; чек {loc["avg_ticket"]:,} руб.

OSM НЕДОСТУПЕН. Оцени все факторы самостоятельно на основе адреса и своих знаний о городе.
"""

    batch = call_batch_ai_full(
        client=client,
        model=model,
        system_prompt=build_ai_full_system_prompt(),
        user_prompt=user_prompt,
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


def similarity_to_reference(target: dict, reference: dict) -> float:
    a = profile_vector(target)
    b = profile_vector(reference)
    weights = np.array([FACTOR_GLOBAL_WEIGHT.get(f, 0) for f in FACTOR_KEYS], dtype=float)
    total_w = np.sum(weights)
    if total_w <= 0:
        return 0.0
    distance = np.sum(np.abs(a - b) * weights) / total_w
    return round(clamp(100.0 - distance), 1)


def similarity_debug(target: dict, reference: dict, ref_name: str) -> Tuple[float, List[Tuple[str, float, float, float]]]:
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
    successful = [r for r in benchmark_rows if r.get("status") == "успешный"]
    weak = [r for r in benchmark_rows if r.get("status") == "слабый"]

    success_similarity = []
    for r in successful:
        sim, debug = similarity_debug(target_profile, r.get("profile", {}), r.get("address", ""))
        success_similarity.append((r.get("address", ""), sim, debug))

    weak_similarity = []
    for r in weak:
        sim, debug = similarity_debug(target_profile, r.get("profile", {}), r.get("address", ""))
        weak_similarity.append((r.get("address", ""), sim, debug))

    success_similarity.sort(key=lambda x: x[1], reverse=True)
    weak_similarity.sort(key=lambda x: x[1], reverse=True)

    successful_centroid = group_centroid([r.get("profile", {}) for r in successful])
    weak_centroid = group_centroid([r.get("profile", {}) for r in weak])

    to_success = similarity_to_reference(target_profile, successful_centroid) if successful_centroid else 0.0
    to_weak = similarity_to_reference(target_profile, weak_centroid) if weak_centroid else 0.0

    return {
        "success_similarity": [(a, b) for a, b, _ in success_similarity],
        "weak_similarity": [(a, b) for a, b, _ in weak_similarity],
        "success_debug": success_similarity,
        "weak_debug": weak_similarity,
        "successful_centroid_similarity": to_success,
        "weak_centroid_similarity": to_weak,
        "benchmark_gap": round(to_success - to_weak, 1),
    }


# ==============================================================================
# HARD RULES  (OSM-зависимые только при доступном OSM)
# ==============================================================================
def calculate_hard_barriers(full_profile: dict, osm: dict, params: dict) -> List[str]:
    barriers = []
    c = osm.get("counts", {})
    osm_available = osm.get("available", False)

    # Параметры локации — всегда
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

    # OSM-зависимые — только если OSM реально доступен (и не пустой)
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
        barriers.append("ℹ️ OSM-данные недоступны — барьеры по парковке, транспорту и застройке не проверены (оценены AI или нейтральные 50).")

    return barriers


def apply_hard_penalties(absolute_score: float, full_profile: dict, barriers: List[str], params: dict, osm_available: bool) -> Tuple[float, float]:
    penalty = 0.0
    # Параметры локации — всегда
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

    # OSM-зависимые — только при доступном OSM
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
def resolve_coordinates(address: str) -> Tuple[Optional[float], Optional[float]]:
    address_lower = address.lower()
    if "энгельса" in address_lower and "екатеринбург" in address_lower:
        return 56.8339, 60.6211
    if "молодогвардейцев" in address_lower and "челябинск" in address_lower:
        return 55.1764, 61.3708
    return get_exact_coordinates(address)


def run_full_analysis(
    api_key: str,
    model: str,
    address: str,
    params: dict,
    target_age: float,
    share_female: float,
    avg_ticket: int,
    clinic_hours: str,
    status_callback=None,
) -> dict:
    target_lat, target_lon = resolve_coordinates(address)
    if target_lat is None or target_lon is None:
        raise ValueError("Не удалось определить координаты. Проверьте адрес.")

    target_loc = {
        "key": "target",
        "address": address,
        "lat": target_lat,
        "lon": target_lon,
        "params": params,
        "target_age": target_age,
        "share_female": share_female,
        "avg_ticket": avg_ticket,
        "clinic_hours": clinic_hours,
    }

    # 1. OSM только для target (1 запрос, 5 сек)
    if status_callback:
        status_callback("1/3", "Собираю OSM-данные для target (таймаут 5 сек)…")
    target_osm = collect_osm_context(target_lat, target_lon)
    osm_target_available = target_osm.get("available", False)

    # 2. AI
    if status_callback:
        status_callback("2/3", "Запрашиваю AI-оценку…")

    ai_failed = False
    target_profile = {}

    if osm_target_available:
        # Режим 1: OSM доступен — AI оценивает 9 факторов
        osm_scores = osm_to_factor_scores(target_osm)
        try:
            ai_profiles = generate_profiles_batch_cached(
                api_key=api_key,
                model=model.strip(),
                locations_json=json.dumps([target_loc], ensure_ascii=False, sort_keys=True),
                osm_json=json.dumps({"target": target_osm}, ensure_ascii=False, sort_keys=True),
            )
            ai = ai_profiles.get("target", make_default_ai_profile())
        except Exception as exc:
            ai_failed = True
            ai = make_default_ai_profile()

        target_profile = dict(osm_scores)
        for ai_key in ["income_fit", "age_fit", "gender_fit", "family_profile", "daytime_balance",
                       "competitor_strength", "market_gap", "noise_safety", "traffic_quality",
                       "profile_confidence", "evidence_quality"]:
            target_profile[ai_key] = ai.get(ai_key, 50)
    else:
        # Режим 2: OSM недоступен — AI оценивает ВСЕ 21 фактор
        try:
            full_ai = generate_full_profile_cached(
                api_key=api_key,
                model=model.strip(),
                location_json=json.dumps(target_loc, ensure_ascii=False, sort_keys=True),
            )
            target_profile = dict(full_ai)
        except Exception as exc:
            ai_failed = True
            target_profile = make_default_full_profile()

    # location_param_score — всегда из правил
    loc_score, _ = compute_location_param_score(params)
    target_profile["location_param_score"] = loc_score

    target_ai = {k: target_profile.get(k, 50) for k in [
        "income_fit", "age_fit", "gender_fit", "family_profile", "daytime_balance",
        "competitor_strength", "market_gap", "noise_safety", "traffic_quality",
        "profile_confidence", "evidence_quality"
    ]}

    # 3. Scoring
    block_scores = compute_block_scores(target_profile)
    absolute_base = compute_absolute_score(block_scores)

    hard_barriers = calculate_hard_barriers(target_profile, target_osm, params)
    absolute_final, hard_penalty = apply_hard_penalties(absolute_base, target_profile, hard_barriers, params, osm_target_available)

    # 4. Benchmark (статические профили)
    benchmark_rows = []
    for row in DATA_CLINICS:
        benchmark_rows.append({
            "address": row["address"],
            "status": row["status"],
            "profile": BENCHMARK_PROFILES[row["key"]],
        })

    benchmark = benchmark_analysis(target_profile, benchmark_rows)
    # Benchmark валиден всегда, т.к. эталоны статические и target оценён (OSM или AI)
    benchmark_valid = True

    if status_callback:
        status_callback("3/3", "Расчёт benchmark…")

    benchmark_component = (
        benchmark["successful_centroid_similarity"] * 0.60 +
        clamp(50 + benchmark["benchmark_gap"] / 2) * 0.40
    )
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

    _, applied_penalties = compute_location_param_score(params)

    return {
        "address": address,
        "params": params,
        "applied_penalties": applied_penalties,
        "latitude": target_lat,
        "longitude": target_lon,
        "profile": target_profile,
        "block_scores": block_scores,
        "osm_context": target_osm,
        "osm_target_available": osm_target_available,
        "absolute_base": absolute_base,
        "absolute_score": absolute_final,
        "hard_penalty": hard_penalty,
        "hard_barriers": hard_barriers,
        "confidence": confidence,
        "benchmark": benchmark,
        "benchmark_valid": benchmark_valid,
        "benchmark_rows": benchmark_rows,
        "final_score": final_score,
        "verdict": verdict,
        "model": model,
        "ai_failed": ai_failed,
    }


# ==============================================================================
# UI — ВВОД
# ==============================================================================
st.divider()

st.subheader("🤖 Настройки AI")
model = st.text_input("Модель OpenAI", value=DEFAULT_MODEL,
    help="Фиксированная модель для стабильности. Если нет GPT-5.1 — укажите gpt-4o.")

st.subheader("🏢 Параметры объекта (критично для оценки)")
st.caption("Отметьте все, что соответствует вашему объекту. Каждый параметр влияет на итоговый score.")

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
        "✅ Первая линия (главная улица, видно с дороги)",
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

clinic_hours = st.text_input("Часы работы", value="08:00–20:00 будни, 09:00–18:00 выходные")

st.subheader("📍 Адрес")
address = st.text_input("Адрес объекта", value="Екатеринбург, Энгельса, 36",
    placeholder="Например: Екатеринбург, Энгельса, 36")

st.divider()
col_a, col_b = st.columns([2, 1])
with col_a:
    run_analysis = st.button("🔍 Запустить анализ", type="primary", use_container_width=True)
with col_b:
    clear_cache = st.button("♻️ Сбросить кэш", use_container_width=True)

if clear_cache:
    st.cache_data.clear()
    st.session_state.pop("last_result", None)
    st.rerun()

# ==============================================================================
# API KEY
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
# RUN
# ==============================================================================
if run_analysis:
    if not address.strip():
        st.error("Адрес не должен быть пустым.")
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
            target_age=float(target_age),
            share_female=float(share_female),
            avg_ticket=int(avg_ticket),
            clinic_hours=clinic_hours.strip(),
            status_callback=update_status,
        )
        st.session_state.last_result = result
        progress_box.success("✅ Анализ завершён.")
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

    if not osm_target_available:
        st.warning("🤖 OSM недоступен — все факторы оценены AI на основе адреса и общих знаний.")
    if ai_failed:
        st.error("🚨 OpenAI недоступен. Использованы нейтральные оценки 50. Результат может быть неточным.")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("FINAL SCORE", f"{result.get('final_score', 0)} / 100")
    with m2:
        st.metric("Абсолютное качество", f"{result.get('absolute_score', 0)} / 100")
    with m3:
        if result.get("benchmark_valid"):
            st.metric("Похожесть на успешные", f"{benchmark.get('successful_centroid_similarity', 0)} / 100")
        else:
            st.metric("Похожесть на успешные", "N/A")
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

    st.caption(f"Базовый score: {result.get('absolute_base', 0)}; hard-penalty: −{result.get('hard_penalty', 0)}")

    applied_penalties = result.get("applied_penalties", [])
    if applied_penalties:
        with st.expander("📐 Расчёт параметров локации"):
            st.markdown("База: **100** (идеальные параметры)")
            for name, penalty, desc in applied_penalties:
                st.markdown(f"−**{penalty}** — *{name}*: {desc}")
            st.markdown(f"**Итог: {profile.get('location_param_score', 0):.0f}/100**")

    # BENCHMARK
    st.subheader("🎯 Benchmark")
    with st.expander("Как считается similarity?"):
        st.markdown("""
**Формула:** взвешенная Manhattan distance по 21 фактору.

```
distance = Σ |target_i − benchmark_i| × weight_i  /  Σ weight_i
similarity = 100 − distance
```

- 100% = профили идентичны
- 0% = максимально разные
- Вес каждого фактора = вес_в_блоке × вес_блока
""")

    bm1, bm2, bm3 = st.columns(3)
    success_sim = benchmark.get("success_similarity", [])
    weak_sim = benchmark.get("weak_similarity", [])
    with bm1:
        st.metric("Ближайший успешный", f"{success_sim[0][1]}%" if success_sim else "—")
    with bm2:
        st.metric("Средний успешных", f"{benchmark.get('successful_centroid_similarity', 0)}%")
    with bm3:
        st.metric("Средний слабых", f"{benchmark.get('weak_centroid_similarity', 0)}%")

    st.metric("Benchmark Gap", f"{benchmark.get('benchmark_gap', 0):+.1f}",
        help="Положительный = ближе к успешным, чем к слабым.")

    bc1, bc2 = st.columns(2)
    with bc1:
        st.markdown("#### Успешные эталоны")
        df_s = pd.DataFrame(success_sim, columns=["Объект", "Similarity"])
        if not df_s.empty:
            df_s["Similarity"] = df_s["Similarity"].map(lambda x: f"{x:.1f}%")
            st.dataframe(df_s, use_container_width=True, hide_index=True)
            success_debug = benchmark.get("success_debug", [])
            if success_debug:
                with st.expander("🔍 Разбор similarity (успешные)"):
                    for addr, sim, debug in success_debug:
                        st.markdown(f"**{addr}**: {sim}%")
                        top5 = debug[:5]
                        for factor, t_val, b_val, contrib in top5:
                            st.markdown(f"  • {factor}: target={t_val}, benchmark={b_val}, вклад={contrib}")
    with bc2:
        st.markdown("#### Слабые эталоны")
        df_w = pd.DataFrame(weak_sim, columns=["Объект", "Similarity"])
        if not df_w.empty:
            df_w["Similarity"] = df_w["Similarity"].map(lambda x: f"{x:.1f}%")
            st.dataframe(df_w, use_container_width=True, hide_index=True)

    # BLOCKS
    st.subheader("🧭 Сводка по блокам")
    block_labels = {
        "location_params": "Параметры локации",
        "parking_access": "Парковка и доступность",
        "demand": "Спрос и ЦА",
        "competition": "Конкуренция",
        "medical_eco": "Медицинская экосистема",
        "visibility_env": "Видимость и среда",
    }
    block_df = pd.DataFrame([
        {"Блок": block_labels.get(b, b), "Score": block_scores.get(b, 0), "Вес": f"{BLOCK_WEIGHTS.get(b, 0)*100:.0f}%"}
        for b in BLOCK_WEIGHTS
    ])
    st.dataframe(block_df, use_container_width=True, hide_index=True)

    # HARD BARRIERS
    st.subheader("🚨 Жёсткие барьеры и риски")
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
        rows.append({
            "": status,
            "Блок": block_labels.get(block, block),
            "Фактор": FACTOR_LABEL.get(factor, factor),
            "Score": round(suitability, 1),
            "Источник": src_icon,
        })
    df_f = pd.DataFrame(rows)
    st.dataframe(df_f, use_container_width=True, hide_index=True, height=560)

    # STRENGTHS / RISKS
    st.subheader("💪 Сильные стороны")
    strong = df_f[df_f["Score"] >= 75].head(10)
    if strong.empty:
        st.write("Нет факторов ≥75.")
    else:
        for _, row in strong.iterrows():
            st.markdown(f"🟢 **{row['Фактор']}** — {row['Score']:.0f}/100 ({row['Источник']})")

    st.subheader("⚠️ Ограничения")
    weak_factors = df_f[df_f["Score"] < 50].sort_values("Score").head(12)
    if weak_factors.empty:
        st.success("Нет факторов ниже 50/100.")
    else:
        for _, row in weak_factors.iterrows():
            st.markdown(f"🔴 **{row['Фактор']}** — {row['Score']:.0f}/100 ({row['Источник']})")

    # OSM AUDIT
    st.subheader("🗺️ OSM-аудит")
    osm = result.get("osm_context", {})
    if osm.get("available"):
        st.success(f"OSM доступен. Элементов: {osm.get('raw_count', 0)}.")
        osm_counts = osm.get("counts", {})
        osm_df = pd.DataFrame([{"Показатель": k, "Количество": v} for k, v in osm_counts.items()])
        st.dataframe(osm_df, use_container_width=True, hide_index=True)
    else:
        st.info(f"🤖 OSM недоступен ({osm.get('error', 'unknown')}). Все факторы оценены AI.")

    st.caption(f"Координаты: {result.get('latitude', 0):.6f}, {result.get('longitude', 0):.6f} · Модель: {result.get('model', model)}")
    with st.expander("Показать полный профиль (JSON)"):
        st.json(profile)


# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.header("Сессия")
    st.success("OpenAI API-ключ активен.")
    st.markdown("""
### Архитектура v3.6

**Параметры локации** (15%)
- 5 явных параметров (чекбоксы)
- Каждый — конкретный штраф

**Парковка + Доступность** (20%)
- OSM: парковки, дороги, транспорт
- Если OSM недоступен → AI оценивает сам

**Спрос + ЦА** (20%)
- OSM: плотность жилой застройки
- AI: доходы, возраст, пол, семьи

**Конкуренция** (15%)
- OSM: количество клиник
- AI: сила конкурентов, зазор

**Мед. экосистема** (15%)
- OSM: аптеки, диагностика, больницы
- AI: оценка при отсутствии OSM

**Видимость + Среда** (15%)
- OSM: тип дорог, видимость
- AI: шум, безопасность, пешеходный комфорт

**Hard rules:** штрафы до 50 баллов
- OSM-зависимые только при доступном OSM

**Benchmark:** 4 успешных + 3 слабых
- Статические предвычисленные профили
- Валиден всегда (нет зависимости от внешних API)

**OSM v3.6:**
- Только 1 запрос для target
- Таймаут 5 сек
- Пустой ответ = unavailable → AI fallback

**AI fallback:**
- При отказе OSM: AI оценивает все 21 фактор
- При отказе OpenAI: нейтральные 50
""")
    if st.button("Сбросить OpenAI ключ"):
        st.session_state.clear()
        st.cache_data.clear()
        st.rerun()
    st.caption("Используйте одну модель и не меняйте эталоны без пересчёта.")
