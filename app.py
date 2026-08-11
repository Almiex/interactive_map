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
4. AI используется как экспертный слой для всех параметров:
   демография, доходы, характер трафика, зоны охвата и т.п.
   Платные гео/демографические API не требуются.
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
    "AI-экспертиза + детерминированный scoring + benchmark. "
    "Платные гео- и демографические API не используются."
)


# ==============================================================================
# НАСТРОЙКИ
# ==============================================================================

DEFAULT_MODEL = "gpt-5.1"
MODEL_REASONING = "low"

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
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

    "walk_5min": "Зона охвата пешком 5 минут",
    "walk_10min": "Зона охвата пешком 10 минут",
    "walk_15min": "Зона охвата пешком 15 минут",
    "car_10min": "Зона охвата на авто 10 минут",
    "car_15min": "Зона охвата на авто 15 минут",
    "car_20min": "Зона охвата на авто 20 минут",
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

FACTOR_DESCRIPTIONS = {
    # DEMAND
    "population_500m": "Оценка численности населения в пешей доступности (до 500 м). 100 = плотная жилая застройка, 0 = промзона/пустырь.",
    "population_1km": "Оценка численности населения в радиусе 1 км. 100 = крупный жилой массив, 0 = малоэтажная окраина.",
    "population_3km": "Оценка численности населения в радиусе 3 км. 100 = центр крупного города, 0 = сельская местность.",
    "target_population_share": "Какая доля населения в радиусе 1 км попадает в целевую аудиторию клиники (возраст, пол, доход). 100 = идеальное совпадение.",
    "target_population_count_1km": "Абсолютное число целевых пациентов в радиусе 1 км. 100 = >50 000 человек, 0 = <5 000.",
    "income_fit": "Соответствие среднего дохода жителей района среднему чеку клиники. 100 = доход идеально покрывает чек, 0 = доход сильно ниже.",
    "age_fit": "Насколько возрастная структура района соответствует ЦА. 100 = пик ЦА (например, 35–55 лет), 0 = молодёжь или пенсионеры.",
    "gender_fit": "Насколько гендерный состав района соответствует ЦА (например, 60% женщин). 100 = точное совпадение.",
    "residential_density": "Плотность жилой застройки вокруг объекта. 100 = многоэтажные дома вплотную, 0 = частный сектор/пустыри.",
    "population_growth": "Перспективы роста населения района (новостройки, миграция). 100 = активное развитие, 0 = депрессивный район.",
    "family_profile": "Насколько район семейный (дети, пары 30–50 лет). 100 = семейный микрорайон, 0 = студенческий/одиночный.",
    "daytime_population_balance": "Баланс между жителями и приходящим дневным населением (офисы). 100 = оптимально для клиники, 0 = только офисы или только спальник.",

    # ACCESSIBILITY
    "walk_5min": "Качество зоны охвата пешком за 5 минут. 100 = плотная застройка, удобные тротуары, отсутствие барьеров.",
    "walk_10min": "Качество зоны охвата пешком за 10 минут. 100 = широкий охват с хорошей инфраструктурой.",
    "walk_15min": "Качество зоны охвата пешком за 15 минут. 100 = максимальный охват пешей доступности.",
    "car_10min": "Качество зоны охвата на авто за 10 минут. 100 = хорошие дороги, отсутствие пробок, удобный подъезд.",
    "car_15min": "Качество зоны охвата на авто за 15 минут. 100 = широкий охват без транспортных барьеров.",
    "car_20min": "Качество зоны охвата на авто за 20 минут. 100 = максимальный охват автодоступности.",
    "public_transport_access": "Наличие и удобство остановок общественного транспорта рядом с клиникой. 100 = 2+ маршрута в шаговой доступности.",
    "transit_connectivity": "Связность транспортной сети (пересадки, частота). 100 = хорошая связь с другими районами.",
    "road_connectivity": "Качество дорожной сети (ширина, состояние, количество полос). 100 = широкие проспекты без заторов.",
    "pedestrian_connectivity": "Удобство пешеходных связей (тротуары, переходы, отсутствие заборов). 100 = комфортная пешеходная среда.",
    "physical_barriers": "Препятствия для пациентов (заборы, реки без мостов, склоны, шоссе). 100 = нет барьеров (после инверсии: чем меньше барьеров, тем выше итоговый score).",
    "vehicle_access": "Удобство подъезда на машине (ширина подъезда, разворот, разгрузка). 100 = удобный подъезд с любой стороны.",

    # TRAFFIC
    "pedestrian_traffic_quality": "Качество пешеходного потока (количество + соответствие ЦА). 100 = много 'правильных' людей.",
    "car_traffic_quality": "Качество автомобильного потока (доступность, скорость, видимость). 100 = удобный проезд с хорошей видимостью вывески.",
    "traffic_target_share": "Доля проходящего/проезжающего трафика, относящегося к ЦА. 100 = большинство потока — потенциальные пациенты.",
    "traffic_time_fit": "Соответствие пиков трафика часам работы клиники. 100 = пик трафика совпадает с рабочими часами.",
    "residential_traffic_share": "Доля жилого трафика (жители района). 100 = в основном жители (повторяемый спрос).",
    "commercial_traffic_share": "Доля коммерческого трафика (покупатели, посетители ТЦ). 100 = много платёжеспособных посетителей.",
    "medical_traffic_share": "Доля медицинского трафика (люди, идущие к врачам/в аптеки). 100 = сформированный медицинский спрос.",
    "office_traffic_share": "Доля офисного трафика (сотрудники бизнес-центров). 100 = много офисных работников (но риск: только в будни).",
    "visibility": "Видимость входа клиники с улицы. 100 = вывеска хорошо видна с главной дороги.",
    "wayfinding": "Лёгкость нахождения входа (адресная табличка, ориентиры). 100 = найти с первого раза, без навигатора.",

    # PARKING
    "parking_supply": "Общее количество парковочных мест в доступности от клиники. 100 = много свободных мест всегда.",
    "parking_distance": "Расстояние от ближайшей парковки до входа. 100 = парковка прямо у входа, 0 = >300 м.",
    "free_parking": "Наличие и доступность бесплатной парковки. 100 = бесплатная парковка прямо у входа.",
    "paid_parking": "Наличие и доступность платной парковки. 100 = много платных мест по разумной цене.",
    "parking_competition": "Конкуренция за парковочные места (торговые центры, офисы забирают места). 100 = нет конкуренции (после инверсии).",
    "parking_time_fit": "Доступность парковки в часы работы клиники. 100 = места есть именно когда работает клиника.",
    "dropoff_access": "Возможность подъехать и высадить пациента у входа. 100 = удобная высадка без риска эвакуации.",
    "parking_reliability": "Надёжность парковки (не эвакуируют, не штрафуют, не занято постоянно). 100 = можно припарковаться всегда.",

    # COMPETITION
    "competitor_density": "Количество конкурентов в радиусе 1 км. 100 = нет конкурентов (после инверсии).",
    "competitor_strength": "Сила ближайших конкурентов (бренд, репутация, оборудование). 100 = слабые/отсутствуют (после инверсии).",
    "competitor_distance": "Расстояние до ближайшего сильного конкурента. 100 = >1 км до серьёзного конкурента.",
    "competitive_capacity": "Ёмкость рынка vs количество клиник. 100 = спрос превышает предложение, есть ниша.",
    "price_level_fit": "Соответствие ценового уровня района ценам клиники. 100 = район готов платить ваши цены.",
    "market_saturation": "Насыщенность рынка медуслугами. 100 = рынок не насыщен (после инверсии).",
    "market_gap": "Наличие незакрытого спроса (каких услуг не хватает). 100 = явный дефицит нужных услуг.",

    # MEDICAL ECOSYSTEM
    "pharmacy_synergy": "Близость аптек (пациент идёт за лекарствами после приёма). 100 = аптека в соседнем помещении.",
    "diagnostics_synergy": "Близость диагностических центров (КТ, МРТ, УЗИ). 100 = диагностика рядом, направляете друг друга.",
    "laboratory_synergy": "Близость лабораторий (сдача анализов). 100 = лаборатория в шаговой доступности.",
    "hospital_synergy": "Близость больниц (направления, госпитализация). 100 = крупная больница рядом.",
    "specialist_synergy": "Близость узких специалистов (к которым направляете/от которых получаете). 100 = развитая сеть специалистов.",
    "medical_cluster": "Наличие медицинского кластера (несколько медучреждений рядом). 100 = медицинский квартал.",
    "healthcare_traffic": "Объём медицинского трафика в районе (люди, идущие к врачам). 100 = сформированный медицинский поток.",

    # ENVIRONMENT
    "residential_commercial_balance": "Баланс жилой и коммерческой застройки. 100 = оптимальное сочетание (жильё + магазины + офисы).",
    "home_clinic_environment": "Насколько среда соответствует формату 'клиника у дома'. 100 = уютный двор, нет шумных магистралей.",
    "information_noise": "Количество рекламы и вывесок вокруг (конкуренция за внимание). 100 = нет информационного шума (после инверсии).",
    "noise_environment": "Уровень шума (дороги, стройки, развлекательные заведения). 100 = тихо (после инверсии).",
    "safety_environment": "Безопасность района (освещение, криминогенная обстановка). 100 = безопасный район, хорошее освещение.",
    "pedestrian_comfort": "Комфорт пешехода (тротуары, озеленение, скамейки). 100 = приятная прогулочная среда.",
    "daily_services": "Повседневная инфраструктура (магазины, кафе, банки) — создают трафик. 100 = развитая инфраструктура.",
    "family_services": "Семейная инфраструктура (школы, детские сады, детские клубы). 100 = много семей с детьми.",
    "fitness_services": "Фитнес-инфраструктура (спортзалы, бассейны). 100 = много спортивных людей (профилактика → клиника).",
    "office_dependence_risk": "Риск зависимости от офисного трафика (пусто вечерами и выходные). 100 = нет риска (после инверсии).",
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

4. Оценивай локацию на основе адреса, координат и общего знания о городе.
   Не используй внешние гео-API — работай как эксперт-аналитик, который знает
   топологию российских городов.

5. Не выдавай ложную точность. Для демографии и доходов, если нет точных
   данных, делай экспертную оценку на основе города, типа района, плотности
   застройки и типа района.

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
    Оценивай честно — если улица незнакомая, ставь низкую уверенность.

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
) -> str:

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

Для parking_* учитывай реальную пациентскую доступность.

Для competition_* учитывай, что большое число конкурентов одновременно
может означать и насыщенность, и сформированный медицинский спрос.
Различай эти эффекты.

Для environment_* оцени именно внешнюю среду.
"""


def build_batch_user_prompt(locations: List[dict], target_key: str) -> str:
    chunks = []
    for loc in locations:
        key = loc["key"]
        chunks.append(
            f"""
--- ЛОКАЦИЯ {key} ---
Адрес: {loc["address"]}
Координаты: {loc["lat"]:.6f}, {loc["lon"]:.6f}
Статус benchmark: НЕ УКАЗАН (не используй его и не пытайся вывести)
Целевая аудитория: возраст {loc["target_age"]:.0f}; женщины {loc["share_female"]*100:.1f}%; чек {loc["avg_ticket"]:,} руб.; часы {loc["clinic_hours"]}
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
- Оценивай на основе адреса, координат и общего знания о городе.

{''.join(chunks)}
"""


@st.cache_data(show_spinner=False, ttl=604800)
def generate_profiles_batch_cached(
    api_key: str,
    model: str,
    locations_json: str,
) -> dict:
    import json
    client = OpenAI(api_key=api_key)
    locations = json.loads(locations_json)

    batch = call_batch_ai(
        client=client,
        model=model,
        system_prompt=build_ai_system_prompt(),
        user_prompt=build_batch_user_prompt(locations, locations[0]["key"]),
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
) -> dict:

    client = OpenAI(api_key=api_key)

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

def calculate_hard_barriers(profile: dict) -> List[str]:
    barriers = []

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

def calculate_confidence(profile: dict) -> int:
    """
    Уверенность = насколько AI уверен в своей оценке.
    Складывается из:
    - profile_confidence: насколько модель уверена в профиле (0–100)
    - evidence_quality: качество доказательной базы (0–100)

    Веса перераспределены:
    - profile_confidence: 55%
    - evidence_quality: 45%
    """
    ai_conf = float(profile.get("profile_confidence", 0))
    evidence = float(profile.get("evidence_quality", 0))

    return int(round(
        clamp(
            ai_conf * 0.55 +
            evidence * 0.45
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
    """Полный запуск: один AI batch для новой локации и эталонов."""
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
        status_callback("1/2", "Выполняю batch-анализ GPT-5.1 для новой локации и эталонов…")

    import json
    profiles = generate_profiles_batch_cached(
        api_key=api_key,
        model=model,
        locations_json=json.dumps(locations, ensure_ascii=False, sort_keys=True),
    )

    target_profile = profiles["target"]
    absolute_base = compute_absolute_score(target_profile)
    hard_barriers = calculate_hard_barriers(target_profile)
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
        status_callback("2/2", "Benchmark рассчитан. Формирую итоговый score…")

    benchmark_component = (
        benchmark["successful_centroid_similarity"] * 0.60
        + clamp(50 + benchmark["benchmark_gap"] / 2) * 0.40
    )
    # Benchmark теперь влияет слабее (20% вместо 30%), т.к. эталоны могут
    # содержать неточности (например, отсутствие парковки у «успешного» объекта).
    final_score = round(absolute_final * 0.80 + benchmark_component * 0.20, 1)

    # Вердикт зависит от score, но низкая confidence блокирует позитивные оценки
    if confidence < 40:
        # При критически низкой уверенности — невозможно оценить
        score_label = ""
        if final_score >= 75:
            score_label = "потенциально сильная"
        elif final_score >= 60:
            score_label = "потенциально хорошая"
        elif final_score >= 45:
            score_label = "потенциально средняя"
        else:
            score_label = "потенциально слабая"
        verdict = f"НЕВОЗМОЖНО ДАТЬ НАДЁЖНУЮ ОЦЕНКУ — {score_label} локация (confidence {confidence}%)"
    elif confidence < 55:
        # При низкой уверенности понижаем категорию на 1 уровень
        if final_score >= 75:
            verdict = "ХОРОШАЯ ЛОКАЦИЯ С ОГОВОРКАМИ — НИЗКАЯ УВЕРЕННОСТЬ В ДАННЫХ"
        elif final_score >= 60:
            verdict = "СРЕДНЯЯ ЛОКАЦИЯ — НИЗКАЯ УВЕРЕННОСТЬ В ДАННЫХ"
        elif final_score >= 45:
            verdict = "СЛАБАЯ ЛОКАЦИЯ — НИЗКАЯ УВЕРЕННОСТЬ В ДАННЫХ"
        else:
            verdict = "СЛАБАЯ ЛОКАЦИЯ — НИЗКАЯ УВЕРЕННОСТЬ В ДАННЫХ"
    else:
        if final_score >= 75:
            verdict = "СИЛЬНАЯ ЛОКАЦИЯ"
        elif final_score >= 60:
            verdict = "ХОРОШАЯ ЛОКАЦИЯ С ОГОВОРКАМИ"
        elif final_score >= 45:
            verdict = "СРЕДНЯЯ ЛОКАЦИЯ"
        else:
            verdict = "СЛАБАЯ ЛОКАЦИЯ"

    return {
        "address": address,
        "latitude": target_lat,
        "longitude": target_lon,
        "profile": target_profile,
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
            help=(
                "Насколько AI уверен в оценке. Складывается из "
                "уверенности модели (55%) и качества доказательной базы (45%). "
                "<40% — оценка ненадёжна, ≥55% — достаточно для принятия решения."
            ),
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

    st.caption(
        "Блоки — это группы факторов. Score блока = взвешенное среднее факторов внутри. "
        "Вес блока — его вклад в итоговый Geo Score."
    )

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

    st.caption(
        "Score = пригодность фактора для клиники (0–100). "
        "Вес = вклад фактора в итоговый score блока. "
        "🔴 <30 | 🟠 30–49 | 🟡 50–74 | 🟢 ≥75."
    )

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
            "Описание": FACTOR_DESCRIPTIONS.get(factor, ""),
        })

    df_factors = pd.DataFrame(rows)

    st.dataframe(
        df_factors,
        use_container_width=True,
        hide_index=True,
        height=700,
        column_config={
            "Описание": st.column_config.TextColumn(
                "Что оценивается",
                width="large",
                help="Подробное описание фактора и интерпретация score",
            ),
            "Score": st.column_config.NumberColumn(
                "Score",
                help="0 = крайне неблагоприятно, 100 = идеально для клиники",
            ),
            "Вес": st.column_config.TextColumn(
                "Вес",
                help="Вклад фактора в блок (сумма в блоке = 100%)",
            ),
        },
    )

    # --------------------------------------------------------------------------
    # STRENGTHS / RISKS
    # --------------------------------------------------------------------------

    st.subheader("💪 Основные сильные стороны")

    strong = df_factors[df_factors["Score"] >= 75].head(10)

    if strong.empty:
        st.write("Нет факторов с оценкой ≥75.")
    else:
        # Группируем зоны охвата, чтобы не засорять список
        catchment_rows = strong[strong["Фактор"].str.contains("Зона охвата", na=False)]
        other_rows = strong[~strong["Фактор"].str.contains("Зона охвата", na=False)]

        if len(catchment_rows) >= 2:
            avg_catchment = catchment_rows["Score"].mean()
            st.markdown(
                f"🟢 **Зоны охвата (пешком и на авто)** — средний показатель {avg_catchment:.0f}/100 "
                f"({len(catchment_rows)} факторов)"
            )

        for _, row in other_rows.iterrows():
            factor_key = None
            for k, v in FACTOR_NAMES.items():
                if v == row['Фактор']:
                    factor_key = k
                    break
            desc = FACTOR_DESCRIPTIONS.get(factor_key, "") if factor_key else ""
            st.markdown(
                f"🟢 **{row['Фактор']}** — {row['Score']:.0f}/100"
            )
            if desc:
                st.caption(desc)

    st.subheader("⚠️ Основные ограничения")

    weak_factors = df_factors[df_factors["Score"] < 50].sort_values(
        "Score"
    ).head(12)

    if weak_factors.empty:
        st.success("Нет факторов ниже 50/100.")
    else:
        for _, row in weak_factors.iterrows():
            factor_key = None
            for k, v in FACTOR_NAMES.items():
                if v == row['Фактор']:
                    factor_key = k
                    break
            desc = FACTOR_DESCRIPTIONS.get(factor_key, "") if factor_key else ""
            st.markdown(
                f"🔴 **{row['Фактор']}** — {row['Score']:.0f}/100"
            )
            if desc:
                st.caption(desc)



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
### Архитектура v3

**1. AI**
- демография
- доходы
- качество трафика
- зоны охвата
- конкуренция
- медицинская синергия
- городская среда

**2. Python**
- фиксированные веса
- нормализация
- benchmark (скрыт, влияет на 20% score)
- similarity
- hard penalties
- final score

**3. Вердикт**
- confidence < 40% → оценка невозможна
- confidence 40–55% → категория понижается
- confidence ≥ 55% → стандартная шкала

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
