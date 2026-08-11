# -*- coding: utf-8 -*-
"""
GeoMarketing AI — Clinic Location Benchmark v2

Что изменено относительно исходной версии:
1. Модель переведена с gpt-4o-mini на GPT-5.1 с высоким reasoning effort.
2. GPT больше НЕ знает статус эталонной клиники во время профилирования.
   Это устраняет label leakage.
3. Добавлен большой внешний геопрофиль: спрос, catchment, транспорт,
   трафик, парковка, конкуренция, медицинская синергия, окружение,
   барьеры и patient friction.
4. Бесплатный OSM/Overpass используется как фактический слой данных.
   Платные гео/демографические API не требуются.
5. AI используется для тех параметров, которых нет в бесплатном OSM
   (демография, доходы, характер трафика и т.п.).
6. Итог считается Python-кодом, а не GPT:
   - Absolute Geo Score
   - Similarity to Successful
   - Similarity to Weak
   - Benchmark Gap
   - Confidence / Data Quality
7. Для каждого фактора используется единая нормализация.
8. Добавлены hard barriers / no-go risks.
9. Эталонная база анализируется один раз и кэшируется.
10. Добавлен режим стабильности: фиксированная версия модели,
    temperature=0 там, где параметр поддерживается, и одинаковый prompt.
"""

import math
import re
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from openai import OpenAI
from pydantic import BaseModel, Field


# ==============================================================================
# STREAMLIT
# ==============================================================================

st.set_page_config(
    page_title="Геомаркетинговый анализ клиники — Benchmark v2",
    page_icon="📍",
    layout="wide",
)

st.title("📍 Геомаркетинговый анализ локации клиники")
st.caption(
    "AI + бесплатные OSM/Overpass-данные + детерминированный benchmark. "
    "Платные гео- и демографические API не используются."
)


# ==============================================================================
# НАСТРОЙКИ
# ==============================================================================

DEFAULT_MODEL = "gpt-5.1"
MODEL_REASONING = "low"

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

REQUEST_HEADERS = {
    "User-Agent": "ClinicGeoAnalytics/2.0 (geobenchmark; educational/business use)"
}

# Весы агрегированных блоков. GPT их не меняет.
BLOCK_WEIGHTS = {
    "demand": 0.27,
    "accessibility": 0.22,
    "traffic": 0.15,
    "parking": 0.07,
    "competition": 0.11,
    "medical_ecosystem": 0.10,
    "environment": 0.08,
}

# Внутри блоков веса также фиксированы.
FACTOR_WEIGHTS = {
    # DEMAND
    "population_500m": 0.08,
    "population_1km": 0.10,
    "population_3km": 0.08,
    "target_population_share": 0.12,
    "target_population_count_1km": 0.10,
    "income_fit": 0.12,
    "age_fit": 0.10,
    "gender_fit": 0.04,
    "residential_density": 0.08,
    "population_growth": 0.05,
    "family_profile": 0.06,
    "daytime_population_balance": 0.07,

    # ACCESSIBILITY
    "walk_5min": 0.08,
    "walk_10min": 0.10,
    "walk_15min": 0.08,
    "car_10min": 0.12,
    "car_15min": 0.12,
    "car_20min": 0.08,
    "public_transport_access": 0.10,
    "transit_connectivity": 0.06,
    "road_connectivity": 0.06,
    "pedestrian_connectivity": 0.06,
    "physical_barriers": 0.08,
    "vehicle_access": 0.06,

    # TRAFFIC
    "pedestrian_traffic_quality": 0.16,
    "car_traffic_quality": 0.10,
    "traffic_target_share": 0.16,
    "traffic_time_fit": 0.12,
    "residential_traffic_share": 0.10,
    "commercial_traffic_share": 0.07,
    "medical_traffic_share": 0.08,
    "office_traffic_share": 0.05,
    "visibility": 0.10,
    "wayfinding": 0.06,

    # PARKING — веса снижены: парковка перестаёт доминировать в score
    "parking_supply": 0.12,
    "parking_distance": 0.10,
    "free_parking": 0.08,
    "paid_parking": 0.04,
    "parking_competition": 0.10,
    "parking_time_fit": 0.10,
    "dropoff_access": 0.10,
    "parking_reliability": 0.08,

    # COMPETITION
    "competitor_density": 0.16,
    "competitor_strength": 0.18,
    "competitor_distance": 0.10,
    "competitive_capacity": 0.18,
    "price_level_fit": 0.10,
    "market_saturation": 0.16,
    "market_gap": 0.12,

    # MEDICAL ECOSYSTEM
    "pharmacy_synergy": 0.14,
    "diagnostics_synergy": 0.18,
    "laboratory_synergy": 0.14,
    "hospital_synergy": 0.10,
    "specialist_synergy": 0.18,
    "medical_cluster": 0.16,
    "healthcare_traffic": 0.10,

    # ENVIRONMENT
    "residential_commercial_balance": 0.15,
    "home_clinic_environment": 0.16,
    "information_noise": 0.10,
    "noise_environment": 0.08,
    "safety_environment": 0.10,
    "pedestrian_comfort": 0.10,
    "daily_services": 0.10,
    "family_services": 0.07,
    "fitness_services": 0.04,
    "office_dependence_risk": 0.10,
}

FACTOR_BLOCKS = {}

for f in [
    "population_500m", "population_1km", "population_3km",
    "target_population_share", "target_population_count_1km",
    "income_fit", "age_fit", "gender_fit", "residential_density",
    "population_growth", "family_profile", "daytime_population_balance"
]:
    FACTOR_BLOCKS[f] = "demand"

for f in [
    "walk_5min", "walk_10min", "walk_15min", "car_10min", "car_15min",
    "car_20min", "public_transport_access", "transit_connectivity",
    "road_connectivity", "pedestrian_connectivity", "physical_barriers",
    "vehicle_access"
]:
    FACTOR_BLOCKS[f] = "accessibility"

for f in [
    "pedestrian_traffic_quality", "car_traffic_quality",
    "traffic_target_share", "traffic_time_fit", "residential_traffic_share",
    "commercial_traffic_share", "medical_traffic_share",
    "office_traffic_share", "visibility", "wayfinding"
]:
    FACTOR_BLOCKS[f] = "traffic"

for f in [
    "parking_supply", "parking_distance", "free_parking", "paid_parking",
    "parking_competition", "parking_time_fit", "dropoff_access",
    "parking_reliability"
]:
    FACTOR_BLOCKS[f] = "parking"

for f in [
    "competitor_density", "competitor_strength", "competitor_distance",
    "competitive_capacity", "price_level_fit", "market_saturation",
    "market_gap"
]:
    FACTOR_BLOCKS[f] = "competition"

for f in [
    "pharmacy_synergy", "diagnostics_synergy", "laboratory_synergy",
    "hospital_synergy", "specialist_synergy", "medical_cluster",
    "healthcare_traffic"
]:
    FACTOR_BLOCKS[f] = "medical_ecosystem"

for f in [
    "residential_commercial_balance", "home_clinic_environment",
    "information_noise", "noise_environment", "safety_environment",
    "pedestrian_comfort", "daily_services", "family_services",
    "fitness_services", "office_dependence_risk"
]:
    FACTOR_BLOCKS[f] = "environment"

FACTOR_NAMES = {
    "population_500m": "Население 500 м",
    "population_1km": "Население 1 км",
    "population_3km": "Население 3 км",
    "target_population_share": "Доля ЦА",
    "target_population_count_1km": "Численность ЦА в 1 км",
    "income_fit": "Соответствие дохода среднему чеку",
    "age_fit": "Возрастное соответствие ЦА",
    "gender_fit": "Половое соответствие ЦА",
    "residential_density": "Плотность жилой застройки",
    "population_growth": "Потенциал роста населения",
    "family_profile": "Семейный профиль",
    "daytime_population_balance": "Баланс дневного и жилого населения",

    "walk_5min": "Catchment пешком 5 минут",
    "walk_10min": "Catchment пешком 10 минут",
    "walk_15min": "Catchment пешком 15 минут",
    "car_10min": "Catchment на авто 10 минут",
    "car_15min": "Catchment на авто 15 минут",
    "car_20min": "Catchment на авто 20 минут",
    "public_transport_access": "Доступность общественным транспортом",
    "transit_connectivity": "Связность общественного транспорта",
    "road_connectivity": "Автомобильная связность",
    "pedestrian_connectivity": "Пешеходная связность",
    "physical_barriers": "Физические барьеры",
    "vehicle_access": "Удобство автомобильного подъезда",

    "pedestrian_traffic_quality": "Качество пешеходного трафика",
    "car_traffic_quality": "Качество автомобильного трафика",
    "traffic_target_share": "Доля трафика из ЦА",
    "traffic_time_fit": "Соответствие времени трафика режиму клиники",
    "residential_traffic_share": "Доля жилого трафика",
    "commercial_traffic_share": "Доля коммерческого трафика",
    "medical_traffic_share": "Доля медицинского трафика",
    "office_traffic_share": "Доля офисного трафика",
    "visibility": "Видимость",
    "wayfinding": "Навигация к входу",

    "parking_supply": "Парковочная ёмкость",
    "parking_distance": "Расстояние от парковки",
    "free_parking": "Бесплатная парковка",
    "paid_parking": "Платная парковка",
    "parking_competition": "Конкуренция за парковку",
    "parking_time_fit": "Парковка в часы работы клиники",
    "dropoff_access": "Высадка/подъезд пациента",
    "parking_reliability": "Надёжность парковки",

    "competitor_density": "Плотность конкурентов",
    "competitor_strength": "Сила конкурентов",
    "competitor_distance": "Дистанция до конкурентов",
    "competitive_capacity": "Конкурентная ёмкость рынка",
    "price_level_fit": "Соответствие ценового уровня",
    "market_saturation": "Насыщенность рынка",
    "market_gap": "Рыночный зазор",

    "pharmacy_synergy": "Синергия с аптеками",
    "diagnostics_synergy": "Синергия с диагностикой",
    "laboratory_synergy": "Синергия с лабораториями",
    "hospital_synergy": "Синергия с больницами",
    "specialist_synergy": "Синергия со специалистами",
    "medical_cluster": "Медицинский кластер",
    "healthcare_traffic": "Медицинский трафик",

    "residential_commercial_balance": "Баланс жилой/коммерческой среды",
    "home_clinic_environment": "Среда «клиника у дома»",
    "information_noise": "Информационный шум",
    "noise_environment": "Шумовая среда",
    "safety_environment": "Безопасность окружения",
    "pedestrian_comfort": "Комфорт пешехода",
    "daily_services": "Повседневная инфраструктура",
    "family_services": "Семейная инфраструктура",
    "fitness_services": "Фитнес-инфраструктура",
    "office_dependence_risk": "Риск зависимости от офисного трафика",
}

# Для факторов, где «больше» хуже, score инвертируется.
LOW_IS_BAD = {
    "physical_barriers": True,
    "competitor_density": True,
    "competitor_strength": True,
    "market_saturation": True,
    "parking_competition": True,
    "information_noise": True,
    "noise_environment": True,
    "office_dependence_risk": True,
}


# ==============================================================================
# ЭТАЛОННЫЕ ОБЪЕКТЫ
# ==============================================================================

DATA_CLINICS = [
    {
        "address": "Красноярск, ул. 9 Мая, 19а",
        "status": "успешный",
        "latitude": 56.067749,
        "longitude": 92.933822,
    },
    {
        "address": "Красноярск, ул. Ладо Кецховели, 34",
        "status": "успешный",
        "latitude": 56.017160,
        "longitude": 92.813882,
    },
    {
        "address": "Екатеринбург, ул. Советская, 42",
        "status": "успешный",
        "latitude": 56.855058,
        "longitude": 60.639260,
    },
    {
        "address": "Казань, ул. Алексея Козина, 2",
        "status": "успешный",
        "latitude": 55.814523,
        "longitude": 49.141033,
    },
    {
        "address": "Новосибирск, ул. Новогодняя, 23/1",
        "status": "слабый",
        "latitude": 54.987320,
        "longitude": 82.911925,
    },
    {
        "address": "Челябинск, ул. Худякова, 10",
        "status": "слабый",
        "latitude": 55.148154,
        "longitude": 61.365313,
    },
    {
        "address": "Самара, ул. Академика Платонова, 10 корпус 3",
        "status": "слабый",
        "latitude": 53.218579,
        "longitude": 50.176465,
    },
]


# ==============================================================================
# PYDANTIC SCHEMA — AI ГЕОПРОФИЛЬ
# ==============================================================================

class GeoAIProfile(BaseModel):
    population_500m: int = Field(ge=0, le=500000)
    population_1km: int = Field(ge=0, le=1000000)
    population_3km: int = Field(ge=0, le=3000000)

    target_population_share: int = Field(ge=0, le=100)
    target_population_count_1km: int = Field(ge=0, le=1000000)

    income_fit: int = Field(ge=0, le=100)
    age_fit: int = Field(ge=0, le=100)
    gender_fit: int = Field(ge=0, le=100)
    residential_density: int = Field(ge=0, le=100)
    population_growth: int = Field(ge=0, le=100)
    family_profile: int = Field(ge=0, le=100)
    daytime_population_balance: int = Field(ge=0, le=100)

    walk_5min: int = Field(ge=0, le=100)
    walk_10min: int = Field(ge=0, le=100)
    walk_15min: int = Field(ge=0, le=100)
    car_10min: int = Field(ge=0, le=100)
    car_15min: int = Field(ge=0, le=100)
    car_20min: int = Field(ge=0, le=100)
    public_transport_access: int = Field(ge=0, le=100)
    transit_connectivity: int = Field(ge=0, le=100)
    road_connectivity: int = Field(ge=0, le=100)
    pedestrian_connectivity: int = Field(ge=0, le=100)
    physical_barriers: int = Field(ge=0, le=100)
    vehicle_access: int = Field(ge=0, le=100)

    pedestrian_traffic_quality: int = Field(ge=0, le=100)
    car_traffic_quality: int = Field(ge=0, le=100)
    traffic_target_share: int = Field(ge=0, le=100)
    traffic_time_fit: int = Field(ge=0, le=100)
    residential_traffic_share: int = Field(ge=0, le=100)
    commercial_traffic_share: int = Field(ge=0, le=100)
    medical_traffic_share: int = Field(ge=0, le=100)
    office_traffic_share: int = Field(ge=0, le=100)
    visibility: int = Field(ge=0, le=100)
    wayfinding: int = Field(ge=0, le=100)

    parking_supply: int = Field(ge=0, le=100)
    parking_distance: int = Field(ge=0, le=100)
    free_parking: int = Field(ge=0, le=100)
    paid_parking: int = Field(ge=0, le=100)
    parking_competition: int = Field(ge=0, le=100)
    parking_time_fit: int = Field(ge=0, le=100)
    dropoff_access: int = Field(ge=0, le=100)
    parking_reliability: int = Field(ge=0, le=100)

    competitor_density: int = Field(ge=0, le=100)
    competitor_strength: int = Field(ge=0, le=100)
    competitor_distance: int = Field(ge=0, le=100)
    competitive_capacity: int = Field(ge=0, le=100)
    price_level_fit: int = Field(ge=0, le=100)
    market_saturation: int = Field(ge=0, le=100)
    market_gap: int = Field(ge=0, le=100)

    pharmacy_synergy: int = Field(ge=0, le=100)
    diagnostics_synergy: int = Field(ge=0, le=100)
    laboratory_synergy: int = Field(ge=0, le=100)
    hospital_synergy: int = Field(ge=0, le=100)
    specialist_synergy: int = Field(ge=0, le=100)
    medical_cluster: int = Field(ge=0, le=100)
    healthcare_traffic: int = Field(ge=0, le=100)

    residential_commercial_balance: int = Field(ge=0, le=100)
    home_clinic_environment: int = Field(ge=0, le=100)
    information_noise: int = Field(ge=0, le=100)
    noise_environment: int = Field(ge=0, le=100)
    safety_environment: int = Field(ge=0, le=100)
    pedestrian_comfort: int = Field(ge=0, le=100)
    daily_services: int = Field(ge=0, le=100)
    family_services: int = Field(ge=0, le=100)
    fitness_services: int = Field(ge=0, le=100)
    office_dependence_risk: int = Field(ge=0, le=100)

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

def call_structured_ai(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> GeoAIProfile:
    """Один структурированный AI-вызов для одной локации."""
    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=GeoAIProfile,
        timeout=60,
    )
    if model.startswith("gpt-5"):
        kwargs["reasoning_effort"] = MODEL_REASONING
    response = client.beta.chat.completions.parse(**kwargs)
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise ValueError("OpenAI не вернул структурированный GeoAIProfile.")
    return parsed


def call_batch_ai(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> GeoProfileBatch:
    """Один AI-вызов сразу для новой локации и всех эталонов.

    Это критически ускоряет запуск: вместо 8 последовательных reasoning
    запросов выполняется один. Статусы successful/weak в prompt не передаются.
    """
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
        raise ValueError("OpenAI не вернул batch GeoProfileBatch.")
    return parsed


# ==============================================================================
# ГЕОКОДИРОВАНИЕ
# ==============================================================================

@st.cache_data(show_spinner=False, ttl=86400)
def get_exact_coordinates(address: str) -> Tuple[Optional[float], Optional[float]]:
    url = NOMINATIM_URL
    params = {
        "q": address,
        "format": "jsonv2",
        "limit": 1,
        "addressdetails": 1,
    }

    try:
        response = requests.get(
            url,
            params=params,
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
# OSM / OVERPASS
# ==============================================================================

def _overpass_request(query: str) -> List[dict]:
    last_error = None

    for url in OVERPASS_URLS:
        try:
            time.sleep(0.10)
            response = requests.post(
                url,
                data={"data": query},
                headers=REQUEST_HEADERS,
                timeout=25,
            )
            response.raise_for_status()
            return response.json().get("elements", [])
        except Exception as exc:
            last_error = exc
            continue

    if last_error:
        raise last_error

    return []


def _count_tags(elements: List[dict], key: str, values: Optional[set] = None) -> int:
    count = 0
    for element in elements:
        tags = element.get("tags", {})
        value = tags.get(key)
        if value is not None and (values is None or value in values):
            count += 1
    return count


@st.cache_data(show_spinner=False, ttl=86400)
def collect_osm_context(lat: float, lon: float) -> dict:
    """
    Бесплатный OSM-аудит.
    Это НЕ заменяет платные traffic/demography APIs, но даёт AI реальные
    географические факты вместо анализа одного только адреса.
    """

    query = f"""
    [out:json][timeout:25];
    (
      nwr(around:500,{lat},{lon})
        ["amenity"~"pharmacy|hospital|clinic|doctors|school|kindergarten|university|fitness_centre|marketplace"];
      nwr(around:1000,{lat},{lon})
        ["amenity"~"pharmacy|hospital|clinic|doctors|school|kindergarten|university|fitness_centre|marketplace"];
      nwr(around:1000,{lat},{lon})
        ["shop"~"supermarket|mall"];
      nwr(around:1000,{lat},{lon})
        ["office"];
      nwr(around:1000,{lat},{lon})
        ["highway"~"primary|secondary|tertiary|residential|service|footway|path"];
      nwr(around:500,{lat},{lon})
        ["highway"~"bus_stop"];
      nwr(around:1000,{lat},{lon})
        ["public_transport"];
      nwr(around:500,{lat},{lon})
        ["amenity"="parking"];
      nwr(around:1000,{lat},{lon})
        ["amenity"="parking"];
      nwr(around:1000,{lat},{lon})
        ["building"="apartments"];
      nwr(around:1000,{lat},{lon})
        ["building"="office"];
      nwr(around:1000,{lat},{lon})
        ["landuse"~"residential|commercial|retail|industrial"];
    );
    out center tags;
    """

    try:
        elements = _overpass_request(query)
    except Exception as exc:
        return {
            "available": False,
            "error": str(exc),
            "counts": {},
            "road_types": {},
            "landuse": {},
            "named_places": [],
        }

    counts = {
        "pharmacy_500m": 0,
        "medical_500m": 0,
        "hospital_500m": 0,
        "pharmacy_1000m": 0,
        "medical_1000m": 0,
        "hospital_1000m": 0,
        "school_1000m": 0,
        "kindergarten_1000m": 0,
        "university_1000m": 0,
        "fitness_1000m": 0,
        "supermarket_1000m": 0,
        "mall_1000m": 0,
        "office_1000m": 0,
        "parking_500m": 0,
        "parking_1000m": 0,
        "bus_stop_500m": 0,
        "public_transport_1000m": 0,
        "apartments_1000m": 0,
        "office_buildings_1000m": 0,
        "primary_1000m": 0,
        "secondary_1000m": 0,
        "tertiary_1000m": 0,
        "residential_roads_1000m": 0,
        "footways_1000m": 0,
        "service_roads_1000m": 0,
    }

    road_types = {}
    landuse = {}

    for element in elements:
        tags = element.get("tags", {})
        amenity = tags.get("amenity")
        shop = tags.get("shop")
        highway = tags.get("highway")
        building = tags.get("building")
        land = tags.get("landuse")

        if amenity == "pharmacy":
            counts["pharmacy_500m"] += 1
            counts["pharmacy_1000m"] += 1

        if amenity in {"clinic", "doctors"}:
            counts["medical_500m"] += 1
            counts["medical_1000m"] += 1

        if amenity == "hospital":
            counts["hospital_500m"] += 1
            counts["hospital_1000m"] += 1

        if amenity == "school":
            counts["school_1000m"] += 1

        if amenity == "kindergarten":
            counts["kindergarten_1000m"] += 1

        if amenity == "university":
            counts["university_1000m"] += 1

        if amenity == "fitness_centre":
            counts["fitness_1000m"] += 1

        if amenity == "parking":
            counts["parking_500m"] += 1
            counts["parking_1000m"] += 1

        if amenity == "bus_stop":
            counts["bus_stop_500m"] += 1

        if "public_transport" in tags:
            counts["public_transport_1000m"] += 1

        if shop == "supermarket":
            counts["supermarket_1000m"] += 1

        if shop == "mall":
            counts["mall_1000m"] += 1

        if "office" in tags:
            counts["office_1000m"] += 1

        if building == "apartments":
            counts["apartments_1000m"] += 1

        if building == "office":
            counts["office_buildings_1000m"] += 1

        if highway:
            road_types[highway] = road_types.get(highway, 0) + 1

            if highway == "primary":
                counts["primary_1000m"] += 1
            elif highway == "secondary":
                counts["secondary_1000m"] += 1
            elif highway == "tertiary":
                counts["tertiary_1000m"] += 1
            elif highway == "residential":
                counts["residential_roads_1000m"] += 1
            elif highway == "footway":
                counts["footways_1000m"] += 1
            elif highway == "service":
                counts["service_roads_1000m"] += 1

        if land:
            landuse[land] = landuse.get(land, 0) + 1

    named_places = []
    for element in elements:
        name = element.get("tags", {}).get("name")
        if name:
            named_places.append(name)

    return {
        "available": True,
        "error": None,
        "counts": counts,
        "road_types": road_types,
        "landuse": landuse,
        "named_places": sorted(set(named_places))[:80],
        "raw_element_count": len(elements),
    }


from concurrent.futures import ThreadPoolExecutor, as_completed


def collect_osm_parallel(locations: List[Tuple[str, float, float]]) -> Dict[str, dict]:
    """Собирает OSM для всех локаций параллельно.

    Ошибка одного адреса не ломает весь анализ: она превращается в
    available=False и будет отражена в confidence.
    """
    result: Dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=min(4, max(1, len(locations)))) as executor:
        futures = {
            executor.submit(collect_osm_context, lat, lon): key
            for key, lat, lon in locations
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                result[key] = future.result()
            except Exception as exc:
                result[key] = {
                    "available": False,
                    "error": str(exc),
                    "counts": {},
                    "road_types": {},
                    "landuse": {},
                    "named_places": [],
                    "raw_element_count": 0,
                }
    return result


# ==============================================================================
# AI PROMPT
# ==============================================================================

def build_ai_system_prompt() -> str:
    return """
Ты — senior geo-marketing analyst, специализирующийся на многофункциональных
частных медицинских клиниках формата «клиника у дома».

Твоя задача — построить единый геопрофиль локации.

КРИТИЧЕСКИЕ ПРАВИЛА:

1. Не используй статус «успешный/слабый». Статус эталонного объекта тебе
   НЕ передаётся и никогда не должен использоваться для оценки.

2. Не оценивай внутренние параметры помещения:
   площадь, этаж, цену аренды/покупки, ремонт, количество кабинетов,
   планировку и т.п.

3. Оценивай ТОЛЬКО внешние параметры:
   население, демографию, доходы, транспорт, трафик, парковку,
   конкурентов, медицинское окружение, городскую среду, барьеры,
   доступность и т.п.

4. OSM-контекст является фактическим наблюдением. Не игнорируй его.
   Если OSM и твоя общая географическая оценка расходятся, отдавай приоритет
   конкретным данным OSM.

5. Не выдавай ложную точность. Для демографии и доходов, если нет точных
   данных, делай экспертную оценку на основе города, типа района, плотности
   застройки и OSM-контекста.

6. Шкала 0–100 означает пригодность фактора для данной клиники.
   Исключение: факторы, название которых содержит «risk», «noise»,
   «competition», «barriers» или «saturation» — для них 100 означает
   максимально благоприятную ситуацию после смысловой интерпретации поля.
   То есть даже для physical_barriers высокий raw score означает много
   барьеров; программный движок затем инвертирует этот фактор.

7. Оценивай не просто наличие трафика, а его КАЧЕСТВО для целевой аудитории.
   20 000 офисных людей 20–29 лет не обязательно лучше 5 000 жителей 35–55.

8. Для клиники особенно важны:
   повторяемость спроса, близость к дому, доступность на машине,
   парковка, понятный подъезд, отсутствие физических барьеров,
   медицинская экосистема и соответствие демографии ЦА.

9. Для catchment используй экспертную оценку доступного спроса, а не просто
   геометрический радиус.

10. Если фактов недостаточно, снижай profile_confidence и evidence_quality.
    НЕ компенсируй недостаток данных искусственно высокими баллами.

11. Все числовые поля должны быть целыми числами в диапазоне 0–100,
    кроме трёх полей population_* и target_population_count_1km,
    которые являются оценками количества людей.
"""


def build_ai_user_prompt(
    address: str,
    lat: float,
    lon: float,
    target_age: float,
    share_female: float,
    avg_ticket: int,
    clinic_hours: str,
    osm_context: dict,
) -> str:

    counts = osm_context.get("counts", {})
    roads = osm_context.get("road_types", {})
    landuse = osm_context.get("landuse", {})
    names = osm_context.get("named_places", [])

    return f"""
АДРЕС:
{address}

КООРДИНАТЫ:
{lat:.6f}, {lon:.6f}

ПОРТРЕТ ЦЕЛЕВОГО ПАЦИЕНТА:
- средний возраст: {target_age:.0f} лет
- доля женщин: {share_female * 100:.1f}%
- ожидаемый средний чек: {avg_ticket:,} руб.
- часы работы клиники: {clinic_hours}

БЕСПЛАТНЫЙ OSM-КОНТЕКСТ:
Данные ниже являются наблюдаемыми объектами OSM. Они не являются
платной статистикой и могут быть неполными.

COUNTS:
{counts}

ROAD TYPES:
{roads}

LANDUSE:
{landuse}

NAMED PLACES:
{names}

ЗАДАЧА:
Построй единый GeoAIProfile.

Для населения и target_population_count используй разумную оценку именно
для данной локации и города. Не притворяйся, что это точная официальная
статистика.

Для факторов traffic_* отдельно оцени:
- абсолютную привлекательность трафика;
- долю трафика, потенциально относящегося к ЦА;
- соответствие трафика часам работы клиники;
- насколько трафик жилой, офисный, коммерческий и медицинский.

Для parking_* учитывай реальную пациентскую доступность, а не только
наличие объекта amenity=parking.

Для competition_* учитывай, что большое число конкурентов одновременно
может означать и насыщенность, и сформированный медицинский спрос.
Различай эти эффекты.

Для environment_* оцени именно внешнюю среду.
"""


def build_batch_user_prompt(locations: List[dict], osm_by_key: Dict[str, dict], target_key: str) -> str:
    import json
    chunks = []
    for loc in locations:
        key = loc["key"]
        osm = osm_by_key.get(key, {})
        chunks.append(
            f"""
--- ЛОКАЦИЯ {key} ---
Адрес: {loc["address"]}
Координаты: {loc["lat"]:.6f}, {loc["lon"]:.6f}
Статус benchmark: НЕ УКАЗАН (не используй его и не пытайся вывести)
Целевая аудитория: возраст {loc["target_age"]:.0f}; женщины {loc["share_female"]*100:.1f}%; чек {loc["avg_ticket"]:,} руб.; часы {loc["clinic_hours"]}
OSM: {json.dumps(osm, ensure_ascii=False, sort_keys=True)}
"""
        )

    return f"""
Нужно независимо построить GeoAIProfile для каждой из {len(locations)} локаций.
Ключ target-локации: {target_key}.

КРИТИЧНО:
- Верни РОВНО один профиль для каждого key.
- Оценивай каждую локацию независимо.
- Не используй статус successful/weak: он намеренно не передан.
- Не сравнивай локации между собой во время профилирования.
- Все поля должны соответствовать схеме GeoAIProfile.
- Если OSM отсутствует, снижай confidence/evidence_quality.

{''.join(chunks)}
"""


@st.cache_data(show_spinner=False, ttl=604800)
def generate_profiles_batch_cached(
    api_key: str,
    model: str,
    locations_json: str,
    osm_json: str,
) -> dict:
    import json
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
# AI PROFILE
# ==============================================================================

@st.cache_data(show_spinner=False, ttl=604800)
def generate_ai_profile_cached(
    api_key: str,
    model: str,
    address: str,
    lat: float,
    lon: float,
    target_age: float,
    share_female: float,
    avg_ticket: int,
    clinic_hours: str,
    osm_context_json: str,
) -> dict:

    import json

    client = OpenAI(api_key=api_key)
    osm_context = json.loads(osm_context_json)

    profile = call_structured_ai(
        client=client,
        model=model,
        system_prompt=build_ai_system_prompt(),
        user_prompt=build_ai_user_prompt(
            address=address,
            lat=lat,
            lon=lon,
            target_age=target_age,
            share_female=share_female,
            avg_ticket=avg_ticket,
            clinic_hours=clinic_hours,
            osm_context=osm_context,
        ),
    )

    return profile.model_dump()


# ==============================================================================
# НОРМАЛИЗАЦИЯ
# ==============================================================================

def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def normalize_population(value: float, low: float, high: float) -> float:
    if high <= low:
        return 50.0
    return clamp((value - low) / (high - low) * 100.0)


def weighted_mean(values: List[Tuple[float, float]]) -> float:
    total_w = sum(w for _, w in values if w > 0)
    if total_w <= 0:
        return 0.0
    return sum(v * w for v, w in values if w > 0) / total_w


def factor_value(profile: dict, factor: str) -> float:
    value = float(profile.get(factor, 0))

    # Численность населения нормализуем в 0–100 отдельно.
    if factor == "population_500m":
        value = normalize_population(value, 5000, 40000)
    elif factor == "population_1km":
        value = normalize_population(value, 10000, 100000)
    elif factor == "population_3km":
        value = normalize_population(value, 50000, 400000)
    elif factor == "target_population_count_1km":
        value = normalize_population(value, 5000, 50000)

    return clamp(value)


def compute_block_scores(profile: dict) -> Dict[str, float]:
    blocks = {}

    for block in BLOCK_WEIGHTS:
        items = []
        for factor, weight in FACTOR_WEIGHTS.items():
            if FACTOR_BLOCKS[factor] != block:
                continue

            value = factor_value(profile, factor)

            # Для «плохих при росте значения» факторов преобразуем raw → suitability.
            if factor in LOW_IS_BAD:
                value = 100.0 - value

            items.append((value, weight))

        blocks[block] = round(weighted_mean(items), 1)

    return blocks


def compute_absolute_score(profile: dict) -> float:
    block_scores = compute_block_scores(profile)

    return round(
        sum(block_scores[b] * BLOCK_WEIGHTS[b] for b in BLOCK_WEIGHTS),
        1,
    )


# ==============================================================================
# BENCHMARK ENGINE
# ==============================================================================

def profile_vector(profile: dict) -> np.ndarray:
    values = []

    for factor in FACTOR_WEIGHTS:
        value = factor_value(profile, factor)

        if factor in LOW_IS_BAD:
            value = 100.0 - value

        values.append(value)

    return np.array(values, dtype=float)


def similarity_to_reference(
    target: dict,
    reference: dict,
) -> float:
    """
    100 = практически идентичный профиль.
    Используем нормированную взвешенную Manhattan distance,
    чтобы один выброс не уничтожил весь similarity.
    """
    a = profile_vector(target)
    b = profile_vector(reference)

    weights = np.array(
        [FACTOR_WEIGHTS[f] for f in FACTOR_WEIGHTS],
        dtype=float,
    )

    distance = np.sum(np.abs(a - b) * weights) / np.sum(weights)
    return round(clamp(100.0 - distance), 1)


def group_centroid(profiles: List[dict]) -> dict:
    if not profiles:
        return {}

    centroid = {}
    for factor in FACTOR_WEIGHTS:
        vals = [factor_value(p, factor) for p in profiles]
        centroid[factor] = float(np.mean(vals))

    # Для centroid достаточно нормализованных значений.
    return centroid


def benchmark_analysis(
    target_profile: dict,
    benchmark_rows: List[dict],
) -> dict:

    successful = [r for r in benchmark_rows if r["status"] == "успешный"]
    weak = [r for r in benchmark_rows if r["status"] == "слабый"]

    successful_profiles = [r["profile"] for r in successful]
    weak_profiles = [r["profile"] for r in weak]

    success_similarity = [
        (
            r["address"],
            similarity_to_reference(target_profile, r["profile"])
        )
        for r in successful
    ]

    weak_similarity = [
        (
            r["address"],
            similarity_to_reference(target_profile, r["profile"])
        )
        for r in weak
    ]

    success_similarity.sort(key=lambda x: x[1], reverse=True)
    weak_similarity.sort(key=lambda x: x[1], reverse=True)

    successful_centroid = group_centroid(successful_profiles)
    weak_centroid = group_centroid(weak_profiles)

    to_success_centroid = (
        similarity_to_reference(target_profile, successful_centroid)
        if successful_centroid else 0.0
    )

    to_weak_centroid = (
        similarity_to_reference(target_profile, weak_centroid)
        if weak_centroid else 0.0
    )

    return {
        "success_similarity": success_similarity,
        "weak_similarity": weak_similarity,
        "successful_centroid_similarity": to_success_centroid,
        "weak_centroid_similarity": to_weak_centroid,
        "benchmark_gap": round(to_success_centroid - to_weak_centroid, 1),
    }


# ==============================================================================
# HARD RULES
# ==============================================================================

def calculate_hard_barriers(profile: dict, osm_context: dict) -> List[str]:
    barriers = []

    counts = osm_context.get("counts", {})

    if profile.get("physical_barriers", 0) >= 85:
        barriers.append(
            "Высокий уровень физических барьеров между потенциальной ЦА и объектом."
        )

    if profile.get("vehicle_access", 0) <= 20:
        barriers.append(
            "Критически неудобный автомобильный подъезд."
        )

    if profile.get("parking_reliability", 0) <= 20:
        barriers.append(
            "Очень низкая надёжность парковки для пациентов."
        )

    if profile.get("wayfinding", 0) <= 20:
        barriers.append(
            "Слабая навигационная понятность объекта."
        )

    if profile.get("home_clinic_environment", 0) <= 20:
        barriers.append(
            "Среда практически не соответствует формату «клиника у дома»."
        )

    # УСИЛЕННЫЕ барьеры по парковке
    if counts.get("parking_500m", 0) == 0 and profile.get("parking_supply", 0) <= 35:
        barriers.append(
            "OSM не показывает парковку в радиусе 500 м, а AI оценивает "
            "парковочную ёмкость как низкую. Отсутствие парковки — критический барьер."
        )

    if profile.get("parking_supply", 0) <= 15:
        barriers.append(
            "Критически низкая парковочная ёмкость. Пациентам физически некуда припарковаться."
        )

    if profile.get("parking_reliability", 0) <= 10:
        barriers.append(
            "Парковка практически отсутствует или занята постоянно — пациенты не смогут приехать на авто."
        )

    return barriers


def apply_hard_penalties(
    absolute_score: float,
    profile: dict,
    hard_barriers: List[str],
) -> Tuple[float, float]:
    penalty = 0.0

    if profile.get("physical_barriers", 0) >= 90:
        penalty += 12
    elif profile.get("physical_barriers", 0) >= 80:
        penalty += 7

    if profile.get("vehicle_access", 0) <= 15:
        penalty += 10
    elif profile.get("vehicle_access", 0) <= 25:
        penalty += 5

    # Усиленные штрафы за отсутствие/нехватку парковки
    if profile.get("parking_reliability", 0) <= 10:
        penalty += 14
    elif profile.get("parking_reliability", 0) <= 20:
        penalty += 8
    elif profile.get("parking_reliability", 0) <= 30:
        penalty += 4

    if profile.get("parking_supply", 0) <= 15:
        penalty += 14
    elif profile.get("parking_supply", 0) <= 30:
        penalty += 7

    if profile.get("home_clinic_environment", 0) <= 15:
        penalty += 7

    penalty = min(penalty, 35.0)
    final = round(clamp(absolute_score - penalty), 1)

    return final, penalty


# ==============================================================================
# CONFIDENCE
# ==============================================================================

def calculate_confidence(profile: dict, osm_context: dict) -> int:
    ai_conf = float(profile.get("profile_confidence", 0))
    evidence = float(profile.get("evidence_quality", 0))

    osm_quality = 100 if osm_context.get("available") else 35
    raw_count = osm_context.get("raw_element_count", 0)

    if raw_count >= 100:
        osm_quality = min(100, osm_quality + 10)
    elif raw_count < 20:
        osm_quality = max(30, osm_quality - 15)

    return int(round(
        clamp(
            ai_conf * 0.45 +
            evidence * 0.35 +
            osm_quality * 0.20
        )
    ))


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
    target_age: float,
    share_female: float,
    avg_ticket: int,
    clinic_hours: str,
    status_callback=None,
) -> dict:
    """Полный запуск: OSM всех локаций параллельно + один AI batch.

    Это устраняет главную причину зависания старой версии.
    """
    target_lat, target_lon = resolve_coordinates(address)
    if target_lat is None or target_lon is None:
        raise ValueError("Не удалось определить координаты адреса. Проверьте адрес.")

    locations = [{
        "key": "target",
        "address": address,
        "lat": target_lat,
        "lon": target_lon,
        "target_age": target_age,
        "share_female": share_female,
        "avg_ticket": avg_ticket,
        "clinic_hours": clinic_hours,
    }]

    for idx, row in enumerate(DATA_CLINICS, start=1):
        locations.append({
            "key": f"benchmark_{idx}",
            "address": row["address"],
            "lat": row["latitude"],
            "lon": row["longitude"],
            "target_age": target_age,
            "share_female": share_female,
            "avg_ticket": avg_ticket,
            "clinic_hours": clinic_hours,
        })

    if status_callback:
        status_callback("1/3", "Собираю бесплатные OSM-данные для 8 локаций параллельно…")

    osm_locations = [(x["key"], x["lat"], x["lon"]) for x in locations]
    osm_by_key = collect_osm_parallel(osm_locations)

    if status_callback:
        status_callback("2/3", "OSM готов. Выполняю один batch-анализ GPT-5.1 для новой локации и эталонов…")

    import json
    profiles = generate_profiles_batch_cached(
        api_key=api_key,
        model=model,
        locations_json=json.dumps(locations, ensure_ascii=False, sort_keys=True),
        osm_json=json.dumps(osm_by_key, ensure_ascii=False, sort_keys=True),
    )

    target_profile = profiles["target"]
    absolute_base = compute_absolute_score(target_profile)
    hard_barriers = calculate_hard_barriers(target_profile, osm_by_key["target"])
    absolute_final, hard_penalty = apply_hard_penalties(absolute_base, target_profile, hard_barriers)

    benchmark_rows = []
    for idx, row in enumerate(DATA_CLINICS, start=1):
        benchmark_rows.append({
            "address": row["address"],
            "status": row["status"],
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "profile": profiles[f"benchmark_{idx}"],
        })

    benchmark = benchmark_analysis(target_profile, benchmark_rows)

    if status_callback:
        status_callback("3/3", "Benchmark рассчитан. Формирую итоговый score…")

    benchmark_component = (
        benchmark["successful_centroid_similarity"] * 0.60
        + clamp(50 + benchmark["benchmark_gap"] / 2) * 0.40
    )
    # Benchmark теперь влияет слабее (20% вместо 30%), т.к. эталоны могут
    # содержать неточности (например, отсутствие парковки у «успешного» объекта).
    final_score = round(absolute_final * 0.80 + benchmark_component * 0.20, 1)

    if final_score >= 75:
        verdict = "СИЛЬНАЯ ЛОКАЦИЯ"
    elif final_score >= 60:
        verdict = "ХОРОШАЯ ЛОКАЦИЯ С ОГОВОРКАМИ"
    elif final_score >= 45:
        verdict = "СРЕДНЯЯ ЛОКАЦИЯ"
    else:
        verdict = "СЛАБАЯ ЛОКАЦИЯ"

    confidence = calculate_confidence(target_profile, osm_by_key["target"])
    if confidence < 55:
        verdict += " — НИЗКАЯ УВЕРЕННОСТЬ В ДАННЫХ"

    return {
        "address": address,
        "latitude": target_lat,
        "longitude": target_lon,
        "profile": target_profile,
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

st.subheader("🤖 Настройки AI-модели")

model = st.text_input(
    "Модель OpenAI",
    value=DEFAULT_MODEL,
    help=(
        "Для стабильного benchmark лучше использовать фиксированную модель/снимок. "
        "Если ваша организация не имеет доступа к GPT-5.1, укажите доступную "
        "структурированную модель."
    ),
)

st.caption(
    "GPT-5.1 используется как экспертный слой; итоговые баллы и benchmark "
    "считаются Python-кодом."
)

st.subheader("👤 1. Портрет целевого пациента")

col1, col2, col3 = st.columns(3)

with col1:
    target_age = st.number_input(
        "Средний возраст, лет",
        min_value=0,
        max_value=120,
        value=35,
        step=1,
    )

with col2:
    share_female_percent = st.number_input(
        "Доля женщин, %",
        min_value=0.0,
        max_value=100.0,
        value=60.0,
        step=1.0,
    )

with col3:
    avg_ticket = st.number_input(
        "Средний чек, руб.",
        min_value=0,
        max_value=1_000_000,
        value=3500,
        step=100,
    )

clinic_hours = st.text_input(
    "Часы работы клиники",
    value="08:00–20:00 по будням, 09:00–18:00 по выходным",
)

st.subheader("📍 2. Адрес")

address = st.text_input(
    "Адрес объекта",
    value="Екатеринбург, Энгельса, 36",
    placeholder="Например: Екатеринбург, Энгельса, 36",
)

st.divider()

col_a, col_b = st.columns([2, 1])

with col_a:
    run_analysis = st.button(
        "🔍 Запустить расширенный анализ",
        type="primary",
        use_container_width=True,
    )

with col_b:
    clear_cache = st.button(
        "♻️ Сбросить AI/benchmark кэш",
        use_container_width=True,
    )

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
    st.info("Введите OpenAI API-ключ. Он хранится только в текущей сессии.")

    key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
    )

    if st.button("Продолжить", type="primary"):
        if not key.strip():
            st.error("Введите OpenAI API-ключ.")
        else:
            st.session_state.openai_key = key.strip()
            st.rerun()

    st.stop()

client = OpenAI(api_key=st.session_state.openai_key)


# ==============================================================================
# RUN
# ==============================================================================

if run_analysis:

    if not address.strip():
        st.error("Адрес не должен быть пустым.")
        st.stop()

    share_female = share_female_percent / 100.0

    progress_box = st.empty()
    detail_box = st.empty()

    def update_status(step: str, text: str):
        progress_box.info(f"**{step}**  {text}")

    try:
        update_status("START", "Проверяю адрес и запускаю быстрый гео-аудит…")
        result = run_full_analysis(
            api_key=st.session_state.openai_key,
            model=model.strip(),
            address=address.strip(),
            target_age=float(target_age),
            share_female=float(share_female),
            avg_ticket=int(avg_ticket),
            clinic_hours=clinic_hours.strip(),
            status_callback=update_status,
        )
        st.session_state.last_result = result
        progress_box.success("✅ Анализ завершён.")
    except Exception as exc:
        progress_box.error("❌ Анализ завершился ошибкой.")
        st.error(f"Не удалось выполнить анализ: {type(exc).__name__}: {exc}")
        st.exception(exc)


# ==============================================================================
# OUTPUT
# ==============================================================================

if "last_result" in st.session_state:

    result = st.session_state.last_result
    profile = result["profile"]
    benchmark = result["benchmark"]

    st.divider()

    st.subheader("📊 Результат анализа")

    st.markdown(f"### {result['address']}")

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.metric(
            "FINAL GEO SCORE",
            f"{result['final_score']} / 100",
        )

    with metric2:
        st.metric(
            "Абсолютное качество",
            f"{result['absolute_score']} / 100",
        )

    with metric3:
        st.metric(
            "Уверенность",
            f"{result['confidence']}%",
        )

    st.info(result["verdict"])

    st.caption(
        f"Базовый score: {result['absolute_base']}; "
        f"hard-penalty: −{result['hard_penalty']}."
    )

    # --------------------------------------------------------------------------
    # BENCHMARK — скрыт по запросу пользователя
    # --------------------------------------------------------------------------
    # Блок сравнения с эталонными объектами скрыт, т.к. эталонная база
    # содержит неточности (например, отсутствие парковки у «успешного» объекта).
    # Расчёт benchmark_component всё ещё участвует в итоговом score с весом 20%.
    #
    # bm1, bm2, bm3 = st.columns(3)
    # ... (benchmark UI удалён)

    # --------------------------------------------------------------------------
    # BLOCKS
    # --------------------------------------------------------------------------

    st.subheader("🧭 Сводка по блокам")

    block_scores = compute_block_scores(profile)

    block_labels = {
        "demand": "Спрос и ЦА",
        "accessibility": "Доступность",
        "traffic": "Качество трафика",
        "parking": "Парковка",
        "competition": "Конкуренция",
        "medical_ecosystem": "Медицинская синергия",
        "environment": "Среда",
    }

    block_df = pd.DataFrame(
        [
            {
                "Блок": block_labels[b],
                "Score": block_scores[b],
                "Вес": f"{BLOCK_WEIGHTS[b] * 100:.0f}%",
            }
            for b in BLOCK_WEIGHTS
        ]
    )

    st.dataframe(
        block_df,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------------------------
    # HARD BARRIERS
    # --------------------------------------------------------------------------

    st.subheader("🚨 Жёсткие барьеры")

    if result["hard_barriers"]:
        for barrier in result["hard_barriers"]:
            st.error(barrier)
    else:
        st.success("Критических hard-barriers не обнаружено.")

    # --------------------------------------------------------------------------
    # FACTORS
    # --------------------------------------------------------------------------

    st.subheader("🔎 Детализация внешних факторов")

    rows = []

    for factor, weight in FACTOR_WEIGHTS.items():
        raw = factor_value(profile, factor)

        if factor in LOW_IS_BAD:
            suitability = 100.0 - raw
        else:
            suitability = raw

        block = FACTOR_BLOCKS[factor]

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
            "Фактор": FACTOR_NAMES[factor],
            "Score": round(suitability, 1),
            "Вес": f"{weight * 100:.1f}%",
        })

    df_factors = pd.DataFrame(rows)

    st.dataframe(
        df_factors,
        use_container_width=True,
        hide_index=True,
        height=620,
    )

    # --------------------------------------------------------------------------
    # STRENGTHS / RISKS
    # --------------------------------------------------------------------------

    st.subheader("💪 Основные сильные стороны")

    strong = df_factors[df_factors["Score"] >= 75].head(10)

    if strong.empty:
        st.write("Нет факторов с оценкой ≥75.")
    else:
        for _, row in strong.iterrows():
            st.markdown(
                f"🟢 **{row['Фактор']}** — {row['Score']:.0f}/100"
            )

    st.subheader("⚠️ Основные ограничения")

    weak_factors = df_factors[df_factors["Score"] < 50].sort_values(
        "Score"
    ).head(12)

    if weak_factors.empty:
        st.success("Нет факторов ниже 50/100.")
    else:
        for _, row in weak_factors.iterrows():
            st.markdown(
                f"🔴 **{row['Фактор']}** — {row['Score']:.0f}/100"
            )

    # --------------------------------------------------------------------------
    # OSM
    # --------------------------------------------------------------------------

    st.subheader("🗺️ Бесплатный OSM-аудит")

    osm = result["osm_context"]

    if osm.get("available"):
        st.success(
            f"OSM доступен. Получено элементов: "
            f"{osm.get('raw_element_count', 0)}."
        )

        osm_counts = osm.get("counts", {})
        osm_df = pd.DataFrame(
            [
                {"Показатель": k, "Количество": v}
                for k, v in osm_counts.items()
            ]
        )

        st.dataframe(
            osm_df,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning(
            "Overpass временно недоступен. AI-профиль всё равно рассчитан, "
            "но confidence снижен."
        )

    # --------------------------------------------------------------------------
    # METADATA
    # --------------------------------------------------------------------------

    st.caption(
        f"Координаты: {result['latitude']:.6f}, "
        f"{result['longitude']:.6f} · "
        f"Модель: {model} · "
        f"Reasoning: {MODEL_REASONING}"
    )

    with st.expander("Показать AI-профиль целиком"):
        st.json(profile)


# ==============================================================================
# SIDEBAR
# ==============================================================================

with st.sidebar:
    st.header("Сессия")

    st.success("OpenAI API-ключ активен для текущей сессии.")

    st.markdown(
        """
### Архитектура v2

**1. OSM**
- POI
- дороги
- парковки
- жилые/офисные объекты
- транспорт

**2. AI**
- демография
- доходы
- качество трафика
- catchment
- конкуренция
- медицинская синергия
- городская среда

**3. Python**
- фиксированные веса
- нормализация
- benchmark
- similarity
- hard penalties
- final score

**4. Benchmark**
- успешные объекты
- слабые объекты
- centroid
- ближайшие аналоги

Статус эталона НЕ передаётся AI во время
профилирования, чтобы исключить label leakage.
"""
    )

    if st.button("Сбросить OpenAI ключ"):
        st.session_state.clear()
        st.cache_data.clear()
        st.rerun()

    st.caption(
        "Для минимизации разброса используйте одну и ту же модель, "
        "фиксированный prompt и не меняйте эталонную базу без "
        "осознанного пересчёта benchmark."
    )
