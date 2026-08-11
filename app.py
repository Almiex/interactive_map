# -*- coding: utf-8 -*-
"""
GeoMarketing AI — Clinic Location Benchmark v3
Кардинальный пересмотр системы оценки.

Что изменено:
1. Всего 18 факторов вместо 65. Убран шум и коррелирующие метрики.
2. Тип локации — явный параметр (БЦ, 1-я линия, жилой дом и т.д.).
3. Hard no-go rules: штрафы до 50 баллов за критические недостатки.
4. AI оценивает ТОЛЬКО 6-8 факторов (демография, доходы, трафик).
   Всё остальное считается детерминированно из OSM + правил.
5. OSM-данные напрямую трансформируются в score без участия AI.
6. Benchmark сильнее влияет на итог (40%).
7. Адаптировано под Streamlit Cloud.
"""

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    page_title="Геомаркетинг клиники — Benchmark v3",
    page_icon="📍",
    layout="wide",
)

st.title("📍 Геомаркетинговый анализ локации клиники — v3")
st.caption("Детерминированный скоринг + минимальный AI-слой. Платные geo-API не нужны.")

# ==============================================================================
# КОНФИГУРАЦИЯ
# ==============================================================================
DEFAULT_MODEL = "gpt-5.1"
MODEL_REASONING = "low"

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

REQUEST_HEADERS = {
    "User-Agent": "ClinicGeoAnalytics/3.0 (streamlit-cloud; business use)"
}

# --- ВЕСА БЛОКОВ (фиксированы, сумма = 1.0) ---
BLOCK_WEIGHTS = {
    "location_type":   0.15,
    "parking_access":  0.20,
    "demand":          0.20,
    "competition":     0.15,
    "medical_eco":     0.15,
    "visibility_env":  0.15,
}

# --- ФАКТОРЫ ВНУТРИ БЛОКОВ ---
# Каждый фактор: (ключ, блок, вес_в_блоке, тип_оценки)
# тип_оценки: osm | ai | user | rule
FACTORS = [
    ("location_type_score",     "location_type",   1.00, "user"),
    ("parking_proximity",       "parking_access",  0.30, "osm"),
    ("parking_supply",          "parking_access",  0.25, "osm"),
    ("vehicle_access",          "parking_access",  0.25, "osm"),
    ("public_transport",        "parking_access",  0.20, "osm"),
    ("population_density",      "demand",          0.30, "osm"),
    ("income_fit",              "demand",          0.25, "ai"),
    ("age_fit",                 "demand",          0.20, "ai"),
    ("family_profile",          "demand",          0.15, "ai"),
    ("daytime_balance",         "demand",          0.10, "ai"),
    ("competitor_density",      "competition",     0.40, "osm"),
    ("competitor_strength",     "competition",     0.35, "ai"),
    ("market_gap",              "competition",     0.25, "ai"),
    ("pharmacy_synergy",        "medical_eco",     0.25, "osm"),
    ("diagnostics_synergy",     "medical_eco",     0.25, "osm"),
    ("hospital_synergy",        "medical_eco",     0.25, "osm"),
    ("medical_cluster",         "medical_eco",     0.25, "osm"),
    ("visibility",              "visibility_env",  0.35, "osm"),
    ("road_type_fit",           "visibility_env",  0.25, "osm"),
    ("pedestrian_comfort",      "visibility_env",  0.20, "osm"),
    ("noise_safety",            "visibility_env",  0.20, "ai"),
]

FACTOR_KEYS = [f[0] for f in FACTORS]
FACTOR_BLOCK = {f[0]: f[1] for f in FACTORS}
FACTOR_WEIGHT_IN_BLOCK = {f[0]: f[2] for f in FACTORS}
FACTOR_SOURCE = {f[0]: f[3] for f in FACTORS}

# Для факторов, где больше = хуже (raw инвертируется)
LOW_IS_BAD = {"competitor_density"}

# --- ТИПЫ ЛОКАЦИЙ ---
LOCATION_TYPES = {
    "first_line_1f": {
        "label": "Первая линия, 1 этаж, отдельный вход",
        "base_score": 95,
        "description": "Оптимально: видимость, удобный вход, обычно есть парковка.",
    },
    "residential_1f": {
        "label": "1 этаж жилого дома, отдельный вход",
        "base_score": 80,
        "description": "Хорошо: жилой трафик, 'клиника у дома', но парковка может быть ограничена.",
    },
    "residential_basement": {
        "label": "Цоколь/подвал жилого дома",
        "base_score": 45,
        "description": "Проблемы: низкая видимость, пешеходный трафик ниже, возможны барьеры.",
    },
    "mall_inside": {
        "label": "Внутри ТЦ/торгового пассажа",
        "base_score": 60,
        "description": "Парковка есть, но навигация сложная, медицинский трафик не целевой.",
    },
    "bc_1f": {
        "label": "1 этаж бизнес-центра",
        "base_score": 40,
        "description": "Плохо для 'клиники у дома': офисный трафик, проблемы с парковкой, выходные пустые.",
    },
    "bc_upper": {
        "label": "2+ этаж бизнес-центра / офисного здания",
        "base_score": 15,
        "description": "Критично: нет видимости, сложная навигация, офисный трафик, парковка — конкуренция с офисами.",
    },
    "other": {
        "label": "Другое / нестандартное",
        "base_score": 50,
        "description": "Требует ручной оценки.",
    },
}

# --- ЭТАЛОНЫ ---
DATA_CLINICS = [
    {"address": "Красноярск, ул. 9 Мая, 19а",       "status": "успешный", "lat": 56.067749, "lon": 92.933822, "type": "first_line_1f"},
    {"address": "Красноярск, ул. Ладо Кецховели, 34", "status": "успешный", "lat": 56.017160, "lon": 92.813882, "type": "residential_1f"},
    {"address": "Екатеринбург, ул. Советская, 42",   "status": "успешный", "lat": 56.855058, "lon": 60.639260, "type": "first_line_1f"},
    {"address": "Казань, ул. Алексея Козина, 2",     "status": "успешный", "lat": 55.814523, "lon": 49.141033, "type": "residential_1f"},
    {"address": "Новосибирск, ул. Новогодняя, 23/1", "status": "слабый",   "lat": 54.987320, "lon": 82.911925, "type": "bc_upper"},
    {"address": "Челябинск, ул. Худякова, 10",       "status": "слабый",   "lat": 55.148154, "lon": 61.365313, "type": "other"},
    {"address": "Самара, ул. Академика Платонова, 10 корпус 3", "status": "слабый", "lat": 53.218579, "lon": 50.176465, "type": "residential_basement"},
]

# ==============================================================================
# PYDANTIC — ТОЛЬКО AI-ФАКТОРЫ
# ==============================================================================
class GeoAIProfile(BaseModel):
    income_fit: int = Field(ge=0, le=100)
    age_fit: int = Field(ge=0, le=100)
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


# ==============================================================================
# OPENAI
# ==============================================================================
def call_batch_ai(client: OpenAI, model: str, system_prompt: str, user_prompt: str) -> GeoProfileBatch:
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


# ==============================================================================
# ГЕОКОДИРОВАНИЕ
# ==============================================================================
@st.cache_data(show_spinner=False, ttl=86400)
def get_exact_coordinates(address: str) -> Tuple[Optional[float], Optional[float]]:
    try:
        response = requests.get(
            NOMINATIM_URL,
            params={"q": address, "format": "jsonv2", "limit": 1, "addressdetails": 1},
            headers=REQUEST_HEADERS,
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
        if not data:
            return None, None
        return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception:
        return None, None


# ==============================================================================
# OVERPASS / OSM
# ==============================================================================
def _overpass_request(query: str) -> List[dict]:
    last_error = None
    for url in OVERPASS_URLS:
        try:
            time.sleep(0.15)
            response = requests.post(url, data={"data": query}, headers=REQUEST_HEADERS, timeout=30)
            response.raise_for_status()
            return response.json().get("elements", [])
        except Exception as exc:
            last_error = exc
            continue
    if last_error:
        raise last_error
    return []


@st.cache_data(show_spinner=False, ttl=86400)
def collect_osm_context(lat: float, lon: float) -> dict:
    query = f"""
    [out:json][timeout:30];
    (
      nwr(around:300,{lat},{lon})["amenity"~"pharmacy|hospital|clinic|doctors"];
      nwr(around:800,{lat},{lon})["amenity"~"pharmacy|hospital|clinic|doctors|school|kindergarten|university|college"];
      nwr(around:800,{lat},{lon})["shop"~"supermarket|mall|chemist"];
      nwr(around:500,{lat},{lon})["amenity"="parking"];
      nwr(around:1000,{lat},{lon})["amenity"="parking"];
      nwr(around:300,{lat},{lon})["highway"~"bus_stop|platform"];
      nwr(around:800,{lat},{lon})["public_transport"];
      nwr(around:500,{lat},{lon})["building"~"apartments|residential|house|detached"];
      nwr(around:1000,{lat},{lon})["building"~"apartments|residential|house|detached"];
      nwr(around:500,{lat},{lon})["building"~"office|commercial|retail"];
      nwr(around:1000,{lat},{lon})["building"~"office|commercial|retail"];
      nwr(around:800,{lat},{lon})["highway"~"primary|secondary|tertiary|residential|service|living_street|unclassified"];
      nwr(around:500,{lat},{lon})["landuse"~"residential|commercial|retail"];
      nwr(around:300,{lat},{lon})["healthcare"~"centre|clinic|doctor|laboratory|diagnostic"];
      nwr(around:800,{lat},{lon})["healthcare"~"centre|clinic|doctor|laboratory|diagnostic"];
    );
    out center tags;
    """
    try:
        elements = _overpass_request(query)
    except Exception as exc:
        return {"available": False, "error": str(exc), "counts": {}, "roads": {}, "landuse": {}, "buildings": {}}

    counts = {
        "pharmacy_300m": 0, "pharmacy_800m": 0,
        "clinic_300m": 0, "clinic_800m": 0,
        "hospital_300m": 0, "hospital_800m": 0,
        "diag_lab_300m": 0, "diag_lab_800m": 0,
        "school_800m": 0, "kindergarten_800m": 0,
        "supermarket_800m": 0, "mall_800m": 0,
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
        shop = tags.get("shop")
        highway = tags.get("highway")
        building = tags.get("building")
        land = tags.get("landuse")
        healthcare = tags.get("healthcare")

        if amenity == "pharmacy":
            counts["pharmacy_300m"] += 1
            counts["pharmacy_800m"] += 1
        if amenity in ("clinic", "doctors") or healthcare in ("clinic", "doctor", "centre"):
            counts["clinic_300m"] += 1
            counts["clinic_800m"] += 1
        if amenity == "hospital" or healthcare == "hospital":
            counts["hospital_300m"] += 1
            counts["hospital_800m"] += 1
        if healthcare in ("laboratory", "diagnostic"):
            counts["diag_lab_300m"] += 1
            counts["diag_lab_800m"] += 1
        if amenity in ("school", "kindergarten", "university", "college"):
            counts["school_800m"] += 1
        if shop in ("supermarket", "chemist"):
            counts["supermarket_800m"] += 1
        if shop == "mall":
            counts["mall_800m"] += 1
        if amenity == "parking":
            counts["parking_500m"] += 1
            counts["parking_1000m"] += 1
        if highway in ("bus_stop", "platform"):
            counts["bus_stop_300m"] += 1
        if "public_transport" in tags:
            counts["public_transport_800m"] += 1
        if building in ("apartments", "residential", "house", "detached"):
            counts["residential_buildings_500m"] += 1
            counts["residential_buildings_1000m"] += 1
        if building in ("office", "commercial", "retail"):
            counts["office_buildings_500m"] += 1
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

    return {
        "available": True,
        "error": None,
        "counts": counts,
        "roads": roads,
        "landuse": landuse,
        "buildings": buildings,
        "raw_count": len(elements),
    }


def collect_osm_parallel(locations: List[Tuple[str, float, float]]) -> Dict[str, dict]:
    result: Dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=min(4, max(1, len(locations)))) as executor:
        futures = {executor.submit(collect_osm_context, lat, lon): key for key, lat, lon in locations}
        for future in as_completed(futures):
            key = futures[future]
            try:
                result[key] = future.result()
            except Exception as exc:
                result[key] = {"available": False, "error": str(exc), "counts": {}, "roads": {}, "landuse": {}, "buildings": {}}
    return result

# ==============================================================================
# ДЕТЕРМИНИРОВАННЫЙ SCORING ИЗ OSM
# ==============================================================================
def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def osm_to_factor_scores(osm: dict, location_type: str) -> Dict[str, float]:
    c = osm.get("counts", {})
    scores = {}

    # PARKING & ACCESS
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

    # DEMAND
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

    # COMPETITION
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

    # MEDICAL ECOSYSTEM
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

    # VISIBILITY & ENV
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

    loc_info = LOCATION_TYPES.get(location_type, LOCATION_TYPES["other"])
    scores["location_type_score"] = loc_info["base_score"]

    return scores


# ==============================================================================
# AI PROMPT (только 8 факторов)
# ==============================================================================
def build_ai_system_prompt() -> str:
    return """Ты — geo-marketing analyst для частных медицинских клиник формата «клиника у дома».

Твоя задача — оценить ТОЛЬКО следующие факторы по шкале 0–100:
1. income_fit — соответствие дохода населения среднему чеку клиники.
2. age_fit — возрастное соответствие целевой аудитории.
3. family_profile — семейный профиль района (дети, семьи).
4. daytime_balance — баланс дневного и жилого населения.
5. competitor_strength — сила конкурентов (100 = слабые/отсутствуют, 0 = сильные сетевые).
6. market_gap — рыночный зазор (100 = большой незакрытый спрос).
7. noise_safety — шумовая обстановка и безопасность (100 = тихо и безопасно).
8. traffic_quality — качество трафика для ЦА (не количество, а соответствие ЦА).

ПРАВИЛА:
- Используй только переданный адрес, город и OSM-контекст.
- Не оценивай парковку, видимость, доступность — это считается отдельно.
- Не используй статус успешный/слабый (он не передан).
- Делай экспертные оценки. Если данных мало — снижай confidence.
- Все оценки целые числа 0–100.
"""


def build_batch_user_prompt(locations: List[dict], osm_by_key: Dict[str, dict], target_key: str) -> str:
    chunks = []
    for loc in locations:
        key = loc["key"]
        osm = osm_by_key.get(key, {})
        loc_type_label = LOCATION_TYPES.get(loc.get("type", "other"), LOCATION_TYPES["other"])["label"]
        chunks.append(f"""
--- ЛОКАЦИЯ {key} ---
Адрес: {loc["address"]}
Тип локации: {loc_type_label}
Координаты: {loc["lat"]:.6f}, {loc["lon"]:.6f}
ЦА: возраст {loc["target_age"]:.0f}; женщины {loc["share_female"]*100:.0f}%; чек {loc["avg_ticket"]:,} руб.
OSM counts: {json.dumps(osm.get("counts", {}), ensure_ascii=False)}
OSM landuse: {json.dumps(osm.get("landuse", {}), ensure_ascii=False)}
""")
    return f"""Построй GeoAIProfile для КАЖДОЙ локации из списка ниже.
Target-локация: {target_key}

КРИТИЧНО:
- Верни РОВНО по одному профилю на каждый key.
- Оценивай независимо, НЕ сравнивай локации между собой.
- Не используй статус successful/weak (не передан).
- Если OSM пустой — снижай confidence и evidence_quality.

{''.join(chunks)}
"""


@st.cache_data(show_spinner=False, ttl=604800)
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
    result = {item.key: item.profile.model_dump() for item in batch.profiles}
    expected = {loc["key"] for loc in locations}
    missing = expected - set(result)
    if missing:
        raise ValueError(f"OpenAI не вернул профили для: {', '.join(sorted(missing))}")
    return result

# ==============================================================================
# SCORING ENGINE
# ==============================================================================
def compute_block_scores(full_profile: dict) -> Dict[str, float]:
    blocks = {b: [] for b in BLOCK_WEIGHTS}
    for factor in FACTOR_KEYS:
        block = FACTOR_BLOCK[factor]
        weight = FACTOR_WEIGHT_IN_BLOCK[factor]
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
    return round(sum(block_scores[b] * BLOCK_WEIGHTS[b] for b in BLOCK_WEIGHTS), 1)


def profile_vector(full_profile: dict) -> np.ndarray:
    values = []
    for factor in FACTOR_KEYS:
        v = full_profile.get(factor, 0)
        if factor in LOW_IS_BAD:
            v = 100.0 - v
        values.append(v)
    return np.array(values, dtype=float)


def similarity_to_reference(target: dict, reference: dict) -> float:
    a = profile_vector(target)
    b = profile_vector(reference)
    weights = np.array([FACTOR_WEIGHT_IN_BLOCK[f] * BLOCK_WEIGHTS[FACTOR_BLOCK[f]] for f in FACTOR_KEYS], dtype=float)
    distance = np.sum(np.abs(a - b) * weights) / np.sum(weights)
    return round(clamp(100.0 - distance), 1)


def group_centroid(profiles: List[dict]) -> dict:
    if not profiles:
        return {}
    centroid = {}
    for factor in FACTOR_KEYS:
        vals = []
        for p in profiles:
            v = p.get(factor, 0)
            if factor in LOW_IS_BAD:
                v = 100.0 - v
            vals.append(v)
        centroid[factor] = float(np.mean(vals))
    return centroid


def benchmark_analysis(target_profile: dict, benchmark_rows: List[dict]) -> dict:
    successful = [r for r in benchmark_rows if r["status"] == "успешный"]
    weak = [r for r in benchmark_rows if r["status"] == "слабый"]
    success_similarity = [(r["address"], similarity_to_reference(target_profile, r["profile"])) for r in successful]
    weak_similarity = [(r["address"], similarity_to_reference(target_profile, r["profile"])) for r in weak]
    success_similarity.sort(key=lambda x: x[1], reverse=True)
    weak_similarity.sort(key=lambda x: x[1], reverse=True)
    successful_centroid = group_centroid([r["profile"] for r in successful])
    weak_centroid = group_centroid([r["profile"] for r in weak])
    to_success = similarity_to_reference(target_profile, successful_centroid) if successful_centroid else 0.0
    to_weak = similarity_to_reference(target_profile, weak_centroid) if weak_centroid else 0.0
    return {
        "success_similarity": success_similarity,
        "weak_similarity": weak_similarity,
        "successful_centroid_similarity": to_success,
        "weak_centroid_similarity": to_weak,
        "benchmark_gap": round(to_success - to_weak, 1),
    }


# ==============================================================================
# HARD RULES / NO-GO
# ==============================================================================
def calculate_hard_barriers(full_profile: dict, osm: dict, location_type: str) -> List[str]:
    barriers = []
    c = osm.get("counts", {})
    if location_type == "bc_upper":
        barriers.append("🚨 КРИТИЧНО: Объект на 2+ этаже БЦ — нет видимости, сложная навигация, офисный трафик.")
    elif location_type == "bc_1f":
        barriers.append("⚠️ Объект в БЦ — офисный трафик, проблемы с парковкой, выходные пустые.")
    elif location_type == "residential_basement":
        barriers.append("⚠️ Цоколь/подвал — низкая видимость, барьеры для входа.")
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
    return barriers


def apply_hard_penalties(absolute_score: float, full_profile: dict, barriers: List[str], location_type: str) -> Tuple[float, float]:
    penalty = 0.0
    if location_type == "bc_upper":
        penalty += 20
    elif location_type == "bc_1f":
        penalty += 12
    elif location_type == "residential_basement":
        penalty += 10
    elif location_type == "mall_inside":
        penalty += 8
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
# FULL ANALYSIS PIPELINE
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
    location_type: str,
    target_age: float,
    share_female: float,
    avg_ticket: int,
    clinic_hours: str,
    status_callback=None,
) -> dict:
    target_lat, target_lon = resolve_coordinates(address)
    if target_lat is None or target_lon is None:
        raise ValueError("Не удалось определить координаты. Проверьте адрес.")

    locations = [{
        "key": "target",
        "address": address,
        "lat": target_lat,
        "lon": target_lon,
        "type": location_type,
        "target_age": target_age,
        "share_female": share_female,
        "avg_ticket": avg_ticket,
        "clinic_hours": clinic_hours,
    }]
    for idx, row in enumerate(DATA_CLINICS, start=1):
        locations.append({
            "key": f"benchmark_{idx}",
            "address": row["address"],
            "lat": row["lat"],
            "lon": row["lon"],
            "type": row.get("type", "other"),
            "target_age": target_age,
            "share_female": share_female,
            "avg_ticket": avg_ticket,
            "clinic_hours": clinic_hours,
        })

    if status_callback:
        status_callback("1/3", "Собираю OSM-данные для 8 локаций…")
    osm_by_key = collect_osm_parallel([(x["key"], x["lat"], x["lon"]) for x in locations])

    osm_scores_by_key = {}
    for loc in locations:
        osm_scores_by_key[loc["key"]] = osm_to_factor_scores(osm_by_key[loc["key"]], loc.get("type", "other"))

    if status_callback:
        status_callback("2/3", "OSM готов. Запрашиваю AI-оценку (8 факторов)…")

    ai_profiles = generate_profiles_batch_cached(
        api_key=api_key,
        model=model,
        locations_json=json.dumps(locations, ensure_ascii=False, sort_keys=True),
        osm_json=json.dumps(osm_by_key, ensure_ascii=False, sort_keys=True),
    )

    full_profiles = {}
    for loc in locations:
        key = loc["key"]
        full = dict(osm_scores_by_key[key])
        ai = ai_profiles.get(key, {})
        for ai_key in ["income_fit", "age_fit", "family_profile", "daytime_balance",
                       "competitor_strength", "market_gap", "noise_safety", "traffic_quality",
                       "profile_confidence", "evidence_quality"]:
            full[ai_key] = ai.get(ai_key, 50)
        full_profiles[key] = full

    target_profile = full_profiles["target"]
    target_ai = ai_profiles["target"]

    block_scores = compute_block_scores(target_profile)
    absolute_base = compute_absolute_score(block_scores)

    hard_barriers = calculate_hard_barriers(target_profile, osm_by_key["target"], location_type)
    absolute_final, hard_penalty = apply_hard_penalties(absolute_base, target_profile, hard_barriers, location_type)

    benchmark_rows = []
    for idx, row in enumerate(DATA_CLINICS, start=1):
        benchmark_rows.append({
            "address": row["address"],
            "status": row["status"],
            "profile": full_profiles[f"benchmark_{idx}"],
        })
    benchmark = benchmark_analysis(target_profile, benchmark_rows)

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

    confidence = calculate_confidence(target_ai, osm_by_key["target"])
    if confidence < 55:
        verdict += " — НИЗКАЯ УВЕРЕННОСТЬ"

    return {
        "address": address,
        "location_type": location_type,
        "latitude": target_lat,
        "longitude": target_lon,
        "profile": target_profile,
        "block_scores": block_scores,
        "osm_context": osm_by_key["target"],
        "absolute_base": absolute_base,
        "absolute_score": absolute_final,
        "hard_penalty": hard_penalty,
        "hard_barriers": hard_barriers,
        "confidence": confidence,
        "benchmark": benchmark,
        "benchmark_rows": benchmark_rows,
        "final_score": final_score,
        "verdict": verdict,
        "model": model,
    }

# ==============================================================================
# UI — ВВОД
# ==============================================================================
st.divider()

st.subheader("🤖 Настройки AI")
model = st.text_input("Модель OpenAI", value=DEFAULT_MODEL,
    help="Фиксированная модель для стабильности. Если нет GPT-5.1 — укажите gpt-4o.")

st.subheader("🏢 Тип объекта")
loc_type_options = {k: v["label"] for k, v in LOCATION_TYPES.items()}
location_type = st.selectbox(
    "Тип размещения клиники",
    options=list(loc_type_options.keys()),
    format_func=lambda x: loc_type_options[x],
    help="Это КРИТИЧНО влияет на оценку. Объект в БЦ и на 1-й линии получат разные base_score.",
)
with st.expander("Подробнее о типах локаций"):
    for k, v in LOCATION_TYPES.items():
        st.markdown(f"**{v['label']}** — base {v['base_score']}/100. {v['description']}")

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
            location_type=location_type,
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
    profile = result["profile"]
    benchmark = result["benchmark"]
    block_scores = result["block_scores"]

    st.divider()
    st.subheader("📊 Результат анализа")
    st.markdown(f"### {result['address']}")
    st.caption(f"Тип локации: {LOCATION_TYPES[result['location_type']]['label']}")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("FINAL SCORE", f"{result['final_score']} / 100")
    with m2:
        st.metric("Абсолютное качество", f"{result['absolute_score']} / 100")
    with m3:
        st.metric("Похожесть на успешные", f"{benchmark['successful_centroid_similarity']} / 100")
    with m4:
        st.metric("Уверенность", f"{result['confidence']}%")

    if result['final_score'] >= 75:
        st.success(f"### {result['verdict']}")
    elif result['final_score'] >= 60:
        st.info(f"### {result['verdict']}")
    elif result['final_score'] >= 45:
        st.warning(f"### {result['verdict']}")
    else:
        st.error(f"### {result['verdict']}")

    st.caption(f"Базовый score: {result['absolute_base']}; hard-penalty: −{result['hard_penalty']}")

    # BENCHMARK
    st.subheader("🎯 Benchmark")
    bm1, bm2, bm3 = st.columns(3)
    with bm1:
        st.metric("Ближайший успешный", f"{benchmark['success_similarity'][0][1]}%" if benchmark["success_similarity"] else "—")
    with bm2:
        st.metric("Средний успешных", f"{benchmark['successful_centroid_similarity']}%")
    with bm3:
        st.metric("Средний слабых", f"{benchmark['weak_centroid_similarity']}%")

    st.metric("Benchmark Gap", f"{benchmark['benchmark_gap']:+.1f}",
        help="Положительный = ближе к успешным, чем к слабым.")

    bc1, bc2 = st.columns(2)
    with bc1:
        st.markdown("#### Успешные эталоны")
        df_s = pd.DataFrame(benchmark["success_similarity"], columns=["Объект", "Similarity"])
        if not df_s.empty:
            df_s["Similarity"] = df_s["Similarity"].map(lambda x: f"{x:.1f}%")
            st.dataframe(df_s, use_container_width=True, hide_index=True)
    with bc2:
        st.markdown("#### Слабые эталоны")
        df_w = pd.DataFrame(benchmark["weak_similarity"], columns=["Объект", "Similarity"])
        if not df_w.empty:
            df_w["Similarity"] = df_w["Similarity"].map(lambda x: f"{x:.1f}%")
            st.dataframe(df_w, use_container_width=True, hide_index=True)

    # BLOCKS
    st.subheader("🧭 Сводка по блокам")
    block_labels = {
        "location_type": "Тип локации",
        "parking_access": "Парковка и доступность",
        "demand": "Спрос и ЦА",
        "competition": "Конкуренция",
        "medical_eco": "Медицинская экосистема",
        "visibility_env": "Видимость и среда",
    }
    block_df = pd.DataFrame([
        {"Блок": block_labels[b], "Score": block_scores[b], "Вес": f"{BLOCK_WEIGHTS[b]*100:.0f}%"}
        for b in BLOCK_WEIGHTS
    ])
    st.dataframe(block_df, use_container_width=True, hide_index=True)

    # HARD BARRIERS
    st.subheader("🚨 Жёсткие барьеры и риски")
    if result["hard_barriers"]:
        for b in result["hard_barriers"]:
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
        block = FACTOR_BLOCK[factor]
        value = profile.get(factor, 0)
        if factor in LOW_IS_BAD:
            suitability = 100.0 - value
        else:
            suitability = value
        src = FACTOR_SOURCE[factor]
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
            "Блок": block_labels[block],
            "Фактор": factor,
            "Score": round(suitability, 1),
            "Источник": src_icon,
        })
    df_f = pd.DataFrame(rows)
    st.dataframe(df_f, use_container_width=True, hide_index=True, height=520)

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
    osm = result["osm_context"]
    if osm.get("available"):
        st.success(f"OSM доступен. Элементов: {osm.get('raw_count', 0)}.")
        osm_counts = osm.get("counts", {})
        osm_df = pd.DataFrame([{"Показатель": k, "Количество": v} for k, v in osm_counts.items()])
        st.dataframe(osm_df, use_container_width=True, hide_index=True)
    else:
        st.warning("Overpass временно недоступен. AI-профиль рассчитан, но confidence снижен.")

    st.caption(f"Координаты: {result['latitude']:.6f}, {result['longitude']:.6f} · Модель: {model}")
    with st.expander("Показать полный профиль"):
        st.json(profile)


# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.header("Сессия")
    st.success("OpenAI API-ключ активен.")
    st.markdown("""
### Архитектура v3

**1. Тип локации** (15%)
- Пользователь явно выбирает тип
- БЦ/2+ этаж = сильный штраф

**2. Парковка + Доступность** (20%)
- OSM: парковки, дороги, транспорт
- Детерминированные правила

**3. Спрос + ЦА** (20%)
- OSM: плотность жилой застройки
- AI: доходы, возраст, семьи

**4. Конкуренция** (15%)
- OSM: количество клиник
- AI: сила конкурентов, зазор

**5. Мед. экосистема** (15%)
- OSM: аптеки, диагностика, больницы

**6. Видимость + Среда** (15%)
- OSM: тип дорог, видимость
- AI: шум, безопасность

**Hard rules:**
- Штрафы до 50 баллов
- Критерии: парковка, подъезд, тип локации

**Benchmark:**
- 4 успешных + 3 слабых объекта
- Статус НЕ передаётся AI
""")
    if st.button("Сбросить OpenAI ключ"):
        st.session_state.clear()
        st.cache_data.clear()
        st.rerun()
    st.caption("Используйте одну модель и не меняйте эталоны без пересчёта.")
