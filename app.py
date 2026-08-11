# -*- coding: utf-8 -*-

import time
import requests
import pandas as pd
import geopandas as gpd
import numpy as np
import streamlit as st
import h3
import folium
from streamlit_folium import st_folium
from openai import OpenAI
from pydantic import BaseModel, Field
from shapely.geometry import Polygon


# ==============================================================================
# НАСТРОЙКИ STREAMLIT
# ==============================================================================

st.set_page_config(
    page_title="Геомаркетинговый анализ клиники",
    page_icon="📍",
    layout="wide",
)

st.title("📍 Геомаркетинговый анализ локации клиники")
st.caption("OpenAI API-ключ хранится только в текущей сессии и не зашит в код.")


# ==============================================================================
# OPENAI KEY — ЗАПРАШИВАЕМ ОДИН РАЗ ЗА СЕССИЮ
# ==============================================================================

if "openai_key" not in st.session_state:
    st.session_state.openai_key = None

if not st.session_state.openai_key:
    st.info("Введите ваш OpenAI API-ключ. Он не хранится в коде.")
    key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
    )

    if st.button("Продолжить", type="primary"):
        if not key.strip():
            st.error("Введите OpenAI API-ключ.")
            st.stop()

        st.session_state.openai_key = key.strip()
        st.rerun()

    st.stop()


client = OpenAI(api_key=st.session_state.openai_key)


# ==============================================================================
# СХЕМЫ OPENAI
# ==============================================================================

class ComprehensiveGeoSchema(BaseModel):
    first_line_location: int = Field(
        description="Видимость фасада с главной улицы, отсутствие барьеров, отдельный вход (1-10)"
    )
    pedestrian_traffic: int = Field(
        description="Интенсивность пешеходного трафика непосредственно у дверей объекта (1-10)"
    )
    car_traffic: int = Field(
        description="Плотность и объем автомобильного потока на прилегающей улице (1-10)"
    )
    parking_availability_real: int = Field(
        description="РЕАЛЬНАЯ доступность парковки для пациентов (1-10)"
    )
    info_noise_barrier: int = Field(
        description="Уровень информационного шума (1-10)"
    )
    location_vibe_home_friendly: int = Field(
        description="Вайб локации 'уютная клиника у дома' (1-10)"
    )
    bus_stop_300m: int
    pvz_wildberries_ozon_yandex_1000m: int
    shopping_mall_1000m: int
    supermarket_500m: int
    supermarket_1000m: int
    fitness_1000m: int
    business_centre_1000m: int
    pharmacy_500m: int
    clinic_competitor_500m: int
    clinic_competitor_1000m: int
    hospital_1000m: int
    population_500m: int
    population_1000m: int
    population_3000m: int
    building_density_score: int
    average_floors: int
    average_income_rub: int
    local_avg_age: float
    local_share_female: float
    local_share_male: float


class TargetScoreSchema(BaseModel):
    first_line_location_score: int = Field(ge=0, le=100)
    pedestrian_traffic_score: int = Field(ge=0, le=100)
    car_traffic_score: int = Field(ge=0, le=100)
    parking_availability_score: int = Field(ge=0, le=100)
    info_noise_score: int = Field(ge=0, le=100)
    location_vibe_score: int = Field(ge=0, le=100)
    target_audience_match_score: int = Field(ge=0, le=100)
    financial_match_score: int = Field(ge=0, le=100)
    medical_synergy_score: int = Field(ge=0, le=100)


# ==============================================================================
# ЭТАЛОННАЯ БАЗА И КАЛИБРОВКА
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
        "address": "Челябинск, ул.Худякова 10",
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


def fetch_comprehensive_profile_via_gpt(address, lat, lon, status):
    system_prompt = (
        "Ты — ведущая экспертная система ритейл-анализа и геомаркетинга. "
        "Проанализируй локацию. Оцени параметры ритейла и трафика по шкале 1-10, "
        "а также рассчитай количественную демографию. "
        "Особое внимание удели барьерам: если точка находится в офисном БЦ "
        "или деловом сити, то доступность парковки для пациентов "
        "(parking_availability_real) падает до 1-2, уровень инфо-шума "
        "(info_noise_barrier) критический (1-3 балла, вывеска затеряется), "
        "вайб уютной клиники у дома (location_vibe_home_friendly) равен 1-3, "
        "а в трафике идет перекос в сторону молодых мужчин-офисников. "
        f"Учти статус объекта в нашей базе: {status}."
    )

    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Адрес: {address}, {lat}, {lon}",
                },
            ],
            response_format=ComprehensiveGeoSchema,
            timeout=30,
        )

        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError("OpenAI не вернул структурированный ответ.")

        return parsed.model_dump()

    except Exception as e:
        raise RuntimeError(
            f"Не удалось откалибровать эталонную точку {address}: {e}"
        ) from e


@st.cache_data(show_spinner=False)
def run_calibration(api_key):
    """
    Калибровка выполняется один раз для конкретного API-ключа.
    API-ключ не выводится на экран и не сохраняется в коде.
    """
    calibration_client = OpenAI(api_key=api_key)

    def fetch(address, lat, lon, status):
        system_prompt = (
            "Ты — ведущая экспертная система ритейл-анализа и геомаркетинга. "
            "Проанализируй локацию. Оцени параметры ритейла и трафика по шкале 1-10, "
            "а также рассчитай количественную демографию. "
            "Особое внимание удели барьерам: если точка находится в офисном БЦ "
            "или деловом сити, то доступность парковки для пациентов "
            "(parking_availability_real) падает до 1-2, уровень инфо-шума "
            "(info_noise_barrier) критический (1-3 балла), "
            "вайб уютной клиники у дома (location_vibe_home_friendly) равен 1-3, "
            "а в трафике идет перекос в сторону молодых мужчин-офисников. "
            f"Учти статус объекта в нашей базе: {status}."
        )

        response = calibration_client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Адрес: {address}, {lat}, {lon}",
                },
            ],
            response_format=ComprehensiveGeoSchema,
            timeout=30,
        )

        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError("OpenAI не вернул структурированный ответ.")

        return parsed.model_dump()

    df_etalon = pd.DataFrame(DATA_CLINICS)
    collected_profiles = []

    for _, row in df_etalon.iterrows():
        collected_profiles.append(
            fetch(
                row["address"],
                row["latitude"],
                row["longitude"],
                row["status"],
            )
        )

    df_features = pd.DataFrame(collected_profiles)
    df_geoprofile = pd.concat([df_etalon, df_features], axis=1)

    metadata_cols = [
        "address",
        "status",
        "latitude",
        "longitude",
        "local_avg_age",
        "local_share_female",
        "local_share_male",
    ]

    scoring_features = [
        col for col in df_features.columns
        if col not in metadata_cols
    ]

    weights = {}
    for factor in scoring_features:
        weights[factor] = abs(
            df_geoprofile.loc[
                df_geoprofile["status"] == "успешный", factor
            ].mean()
            -
            df_geoprofile.loc[
                df_geoprofile["status"] == "слабый", factor
            ].mean()
        )

    total = sum(weights.values()) + 1e-5
    normalized_weights = {
        key: value / total
        for key, value in weights.items()
    }

    return normalized_weights


# ==============================================================================
# ГЕОКОДИРОВАНИЕ
# ==============================================================================

@st.cache_data(show_spinner=False)
def get_exact_coordinates(address):
    """
    Геокодирование через Nominatim / OpenStreetMap.
    """
    url = "https://nominatim.openstreetmap.org/search"

    headers = {
        "User-Agent": "ClinicGeoAnalytics/1.0",
    }

    params = {
        "q": address,
        "format": "jsonv2",
        "limit": 1,
    }

    try:
        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=15,
        )
        response.raise_for_status()

        data = response.json()

        if not data:
            return None, None

        return float(data[0]["lat"]), float(data[0]["lon"])

    except Exception:
        return None, None


# ==============================================================================
# ЖЕСТКИЙ АУДИТ OSM
# ==============================================================================

def hard_audit_location_via_osm_tags(lat, lon):
    """Программный аудит скрытых барьеров через теги OSM."""

    url = "https://overpass-api.de/api/interpreter"

    query = f"""
    [out:json][timeout:15];
    (
      nwr["office"](around:200,{lat},{lon});
      nwr["highway"="primary"](around:100,{lat},{lon});
      nwr["highway"="secondary"](around:100,{lat},{lon});
    );
    out tags;
    """

    is_office_heavy = False
    is_speed_highway = False

    try:
        time.sleep(0.5)

        response = requests.post(
            url,
            data={"data": query},
            timeout=20,
        )

        if response.status_code == 200:
            elements = response.json().get("elements", [])

            office_count = sum(
                1
                for element in elements
                if "office" in element.get("tags", {})
            )

            if office_count >= 3:
                is_office_heavy = True

            for element in elements:
                highway = element.get("tags", {}).get("highway")

                if highway in ["primary", "secondary"]:
                    is_speed_highway = True

    except Exception:
        pass

    return is_office_heavy, is_speed_highway


# ==============================================================================
# ОСНОВНОЙ АНАЛИЗ
# ==============================================================================

WEIGHTS = {
    "first_line_location_score": 0.10,
    "pedestrian_traffic_score": 0.15,
    "car_traffic_score": 0.05,
    "parking_availability_score": 0.15,
    "info_noise_score": 0.10,
    "location_vibe_score": 0.15,
    "target_audience_match_score": 0.15,
    "financial_match_score": 0.05,
    "medical_synergy_score": 0.10,
}

FACTOR_NAMES = {
    "location_vibe_score": "Вайб «уютная клиника у дома»",
    "parking_availability_score": "Реальная доступность парковки",
    "target_audience_match_score": "Соответствие половозрастной ЦА",
    "pedestrian_traffic_score": "Пешеходный трафик непосредственно у дверей",
    "financial_match_score": "Соответствие доходов среднему чеку",
    "info_noise_score": "Видимость фасада и инфо-шум",
    "first_line_location_score": "Соответствие параметрам первой линии",
    "car_traffic_score": "Интенсивность автомобильного трафика",
    "medical_synergy_score": "Синергия с медицинским окружением",
}


def analyze_location(address, target_age, share_female, avg_ticket):
    # Специальные координаты из исходной логики.
    address_lower = address.lower()

    if "энгельса" in address_lower:
        lat, lon = 56.8339, 60.6211

    elif "молодогвардейцев" in address_lower:
        lat, lon = 55.1764, 61.3708

    else:
        lat, lon = get_exact_coordinates(address)

        if lat is None or lon is None:
            raise ValueError(
                "Не удалось определить координаты адреса. "
                "Проверьте написание адреса."
            )

    # Шаг 1. OSM-аудит.
    is_office_heavy, is_speed_highway = (
        hard_audit_location_via_osm_tags(lat, lon)
    )

    # Шаг 2. Экспертная оценка GPT.
    system_prompt = (
        "Ты — ведущая экспертная система оценки коммерческой недвижимости "
        "под медицинские клиники. "
        "Твоя задача — оценить пригодность локации для открытия уютной, "
        "надежной многофункциональной клиники «у дома». "
        "Оценивай каждый параметр строго по шкале от 0 "
        "(абсолютно непригодно) до 100 (идеально)."
    )

    user_content = (
        f"Адрес объекта для анализа: {address}.\n"
        f"Координаты: {lat}, {lon}.\n"
        f"Портрет целевого пациента клиники:\n"
        f"- Средний возраст: {target_age} лет\n"
        f"- Доля женщин: {share_female * 100:.1f}%\n"
        f"- Ожидаемый средний чек: {avg_ticket} руб.\n\n"
        f"Программный аудит OSM:\n"
        f"- Офисный кластер БЦ: {is_office_heavy}\n"
        f"- Скоростная магистраль рядом: {is_speed_highway}\n"
    )

    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        response_format=TargetScoreSchema,
        timeout=30,
    )

    scores = response.choices[0].message.parsed

    if scores is None:
        raise ValueError("OpenAI не вернул структурированный результат.")

    # Корректировки барьеров из исходной логики.
    if is_office_heavy or "энгельса" in address_lower:
        scores.parking_availability_score = 5
        scores.location_vibe_score = 15

    if is_speed_highway or "энгельса" in address_lower:
        scores.pedestrian_traffic_score = 15
        scores.info_noise_score = min(
            scores.info_noise_score,
            30,
        )

    final_score = sum(
        getattr(scores, factor) * weight
        for factor, weight in WEIGHTS.items()
    )

    final_score = round(final_score, 1)

    if final_score >= 70:
        verdict = (
            "СИЛЬНАЯ ЛОКАЦИЯ "
            "(Полное соответствие вайбу клиники, портрету ЦА и парковкам)"
        )
    elif final_score >= 45:
        verdict = (
            "СРЕДНЯЯ ЛОКАЦИЯ "
            "(Присутствуют критические инфраструктурные риски)"
        )
    else:
        verdict = (
            "СЛАБАЯ ЛОКАЦИЯ "
            "(Критическое несоответствие вайбу, парковкам "
            "или пешеходному трафику!)"
        )

    return {
        "address": address,
        "latitude": lat,
        "longitude": lon,
        "is_office_heavy": is_office_heavy,
        "is_speed_highway": is_speed_highway,
        "scores": scores,
        "final_score": final_score,
        "verdict": verdict,
    }


# ==============================================================================
# ИНТЕРФЕЙС
# ==============================================================================

st.divider()

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


st.subheader("📍 2. Адрес")

address = st.text_input(
    "Адрес объекта",
    value="Екатеринбург, Энгельса, 36",
    placeholder="Например: Екатеринбург, Энгельса, 36",
)

st.divider()

# ==============================================================================
# КНОПКА АНАЛИЗА
# ==============================================================================

if st.button("🔍 Запустить анализ", type="primary", use_container_width=True):

    if not address.strip():
        st.error("Адрес не должен быть пустым.")
        st.stop()

    share_female = share_female_percent / 100.0

    with st.spinner("Проводится геомаркетинговый анализ..."):

        try:
            result = analyze_location(
                address=address.strip(),
                target_age=float(target_age),
                share_female=float(share_female),
                avg_ticket=int(avg_ticket),
            )

            st.session_state.last_result = result

        except Exception as e:
            st.error(f"Ошибка анализа: {e}")
            st.stop()


# ==============================================================================
# ВЫВОД РЕЗУЛЬТАТА
# ==============================================================================

if "last_result" in st.session_state:

    result = st.session_state.last_result
    scores = result["scores"]

    st.divider()
    st.subheader("📊 Результат анализа")

    st.markdown(
        f"### {result['address']}"
    )

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.metric(
            "LOCATION SCORE",
            f"{result['final_score']} / 100",
        )

    with metric2:
        st.metric(
            "Офисный кластер БЦ",
            "Да" if result["is_office_heavy"] else "Нет",
        )

    with metric3:
        st.metric(
            "Скоростная магистраль",
            "Да" if result["is_speed_highway"] else "Нет",
        )

    st.info(result["verdict"])

    st.subheader("Детализация внутренних оценок")

    rows = []

    for factor, label in FACTOR_NAMES.items():
        value = getattr(scores, factor)

        if value >= 70:
            status = "🟢"
        elif value > 30:
            status = "🟡"
        else:
            status = "🔴"

        rows.append(
            {
                "": status,
                "Фактор": label,
                "Оценка": value,
                "Вес": f"{WEIGHTS[factor] * 100:.0f}%",
            }
        )

    df_scores = pd.DataFrame(rows)

    st.dataframe(
        df_scores,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("💪 Основные сильные стороны")

    advantages = []

    for factor, label in FACTOR_NAMES.items():
        value = getattr(scores, factor)

        if value >= 85:
            advantages.append(
                f"**{label}** — {value}/100"
            )

    if advantages:
        for item in advantages:
            st.markdown(f"🟢 {item}")
    else:
        st.write(
            "Явно выраженных сверхвысоких показателей (>85) не обнаружено."
        )

    st.subheader("⚠️ Выявленные барьеры, риски и ограничения")

    risks = []

    for factor, label in FACTOR_NAMES.items():
        value = getattr(scores, factor)

        if value <= 30:
            risks.append(
                f"🔴 **{label}** — {value}/100: "
                "критически низкое значение. "
                "Потенциально блокирующий фактор."
            )

        elif value < 75:
            risks.append(
                f"🟡 **{label}** — {value}/100: "
                "ограничение по фактору, требует контроля "
                "или дополнительного аудита."
            )

    if risks:
        for item in risks:
            st.markdown(item)
    else:
        st.success(
            "Критических рисков и инфраструктурных барьеров не обнаружено."
        )

    st.caption(
        f"Координаты объекта: {result['latitude']:.6f}, "
        f"{result['longitude']:.6f}"
    )
#========
#==КАРТА
#========

import streamlit as st
import pandas as pd
import numpy as np
import h3
import folium
import requests
import pyproj
from streamlit_folium import st_folium
from shapely.geometry import Polygon
from shapely.ops import transform

# ==============================================================================
# 1. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ И СБОР ДАННЫХ
# ==============================================================================

@st.cache_data(show_spinner=False, ttl=600)
def fetch_osm_data_fast(lat, lon, radius_m=2000):
    """Быстрый и надежный сбор данных через Overpass API"""
    url = "https://openstreetmap.fr"
    # Запрашиваем ВСЕ здания (включая building=yes) и инфраструктуру
    query = f"""
    [out:json][timeout:25];
    (
      way["building"](around:{radius_m},{lat},{lon});
      relation["building"](around:{radius_m},{lat},{lon});
      way["highway"](around:{radius_m},{lat},{lon});
      node["amenity"~"clinic|hospital|doctors|pharmacy|dentist"](around:{radius_m},{lat},{lon});
      node["shop"~"mall|supermarket|car|jewelry|boutique|beauty"](around:{radius_m},{lat},{lon});
      node["office"](around:{radius_m},{lat},{lon});
    );
    out center geom;
    """
    try:
        r = requests.post(url, data={"data": query}, timeout=25)
        r.raise_for_status()
        return r.json().get("elements", [])
    except Exception:
        try:
            r = requests.post("https://overpass-api.de", data={"data": query}, timeout=25)
            return r.json().get("elements", [])
        except Exception:
            return []

def _calculate_polygon_area_3857(geometry_nodes):
    """Перевод координат Overpass в метры (EPSG:3857) и точный расчет площади фундамента"""
    if not geometry_nodes or len(geometry_nodes) < 3:
        return 0.0
    try:
        coords = [(pt['lon'], pt['lat']) for pt in geometry_nodes]
        poly = Polygon(coords)
        
        # Проекция Меркатора для перевода градусов в метры
        wgs84 = pyproj.CRS('EPSG:4326')
        utm = pyproj.CRS('EPSG:3857')
        project = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True).transform
        poly_meters = transform(project, poly)
        
        return poly_meters.area
    except Exception:
        return 0.0

def build_hex_grid_v9(lat, lon, radius_m=2000):
    """Построение сетки H3 с разрешением 9 (шаг ~182м) в радиусе 2км"""
    try:
        center_hex = h3.latlng_to_cell(lat, lon, 9)
    except AttributeError:
        center_hex = h3.geo_to_h3(lat, lon, 9)
        
    k = int(np.ceil(radius_m / 182)) + 1
    
    try:
        candidates = h3.grid_disk(center_hex, k)
    except AttributeError:
        candidates = h3.k_ring(center_hex, k)
        
    hexes = []
    for h in candidates:
        try:
            hlat, hlon = h3.cell_to_latlng(h)
        except AttributeError:
            hlat, hlon = h3.h3_to_geo(h)
            
        # Формула Гаверсинуса для удержания строгого радиуса 2км
        R = 6371.0
        phi1, phi2 = np.radians(lat), np.radians(hlat)
        dphi = np.radians(hlat - lat)
        dlambda = np.radians(hlon - lon)
        a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
        dist = 2 * R * np.arcsin(np.sqrt(a)) * 1000
        
        if dist <= radius_m:
            hexes.append(h)
    return hexes

# ==============================================================================
# 2. АНАЛИТИКА МЕТРИК
# ==============================================================================

def compute_hex_metrics_v9(hexes, elements):
    data = {h: {"bld_pop": [], "bld_levels": [], "traffic_count": 0, "biz_count": 0, "med_count": 0} for h in hexes}
    
    for el in elements:
        elat = el.get("lat") or el.get("center", {}).get("lat")
        elon = el.get("lon") or el.get("center", {}).get("lon")
        if elat is None or elon is None:
            continue

        try:
            h = h3.latlng_to_cell(float(elat), float(elon), 9)
        except AttributeError:
            h = h3.geo_to_h3(float(elat), float(elon), 9)
            
        if h not in data:
            continue

        tags = el.get("tags", {})
        
        # 1. Логика расчета населения и этажности (из Колаба)
        if "building" in tags:
            b_type = tags.get("building")
            # Исключаем заведомо нежилые промзоны/трансформаторы
            if b_type not in ["industrial", "gargages", "substation", "ruins"]:
                geom_nodes = el.get("geometry", [])
                footprint_area = _calculate_polygon_area_3857(geom_nodes)
                
                # Если геометрия неполная, берем средний след дома в зависимости от типа
                if footprint_area == 0:
                    footprint_area = 600.0 if b_type == "apartments" else 150.0
                
                levels = tags.get('building:levels', None)
                if levels is None or not str(levels).isdigit():
                    levels = 9 if b_type == 'apartments' else 5
                else:
                    levels = int(levels)
                
                # Ваша формула расчета людей из Колаба
                total_living_area = footprint_area * levels
                estimated_people = int(total_living_area / 27)
                if estimated_people < 1:
                    estimated_people = 2
                    
                data[h]["bld_pop"].append(estimated_people)
                data[h]["bld_levels"].append(levels)

        # 2. Подсчет путей трафика
        if "highway" in tags:
            data[h]["traffic_count"] += 1

        # 3. Подсчет медицинских конкурентов
        if tags.get("amenity") in ["clinic", "hospital", "doctors", "pharmacy", "dentist"]:
            data[h]["med_count"] += 1

        # 4. Подсчет коммерческих объектов
        if "shop" in tags or "office" in tags or tags.get("building") in ["commercial", "retail", "office"]:
            data[h]["biz_count"] += 1

    # Собираем физические значения воедино
    metrics = {}
    for h in hexes:
        d = data[h]
        metrics[h] = {
            "population": int(sum(d["bld_pop"])),
            "floors": round(float(np.mean(d["bld_levels"])), 1) if d["bld_levels"] else 0.0,
            "traffic": int(d["traffic_count"]),
            "income": int(d["biz_count"]),
            "competition": int(d["med_count"]),
        }
    return metrics

def _hex_color(value, vmin, vmax, palette):
    if vmax == vmin:
        return palette[0]
    ratio = (value - vmin) / (vmax - vmin)
    idx = int(ratio * (len(palette) - 1))
    idx = max(0, min(idx, len(palette) - 1))
    return palette[idx]

# ==============================================================================
# 3. ИНТЕРФЕЙС И КАРТА СТРИМЛИТ (БЕЗ ТАБЛИЦЫ ЛИДЕРОВ)
# ==============================================================================

st.divider()
st.subheader("🗺️ Гео-аналитика радиуса 2 км (гексагональная сетка OSM)")

if "last_result" in st.session_state:
    result = st.session_state.last_result
    lat, lon = result["latitude"], result["longitude"]

    filter_key = st.selectbox(
        "Выберите активный слой данных:",
        options=["population", "floors", "traffic", "income", "competition"],
        format_func=lambda x: {
            "population": "👥 Плотность населения (чел.)",
            "floors": "🏢 Этажность застройки (этажи)",
            "traffic": "🚗 Пешеходный + авто трафик (пути)",
            "income": "💼 Бизнес-инфраструктура (объекты)",
            "competition": "🏥 Конкуренция медучреждений (объекты)"
        }.get(x, x)
    )

    with st.spinner("Загружаем данные OSM и перестраиваем сетку…"):
        try:
            # Скачиваем через стабильный Overpass
            elements = fetch_osm_data_fast(lat, lon, radius_m=2000)
            grid = build_hex_grid_v9(lat, lon, radius_m=2000)
            hex_metrics = compute_hex_metrics_v9(grid, elements)

            if not hex_metrics:
                st.warning("Не удалось сгенерировать гексагональную сетку.")
            else:
                m = folium.Map(location=[lat, lon], zoom_start=13, tiles="CartoDB positron")
                
                # Круг радиуса анализа
                folium.Circle(
                    location=[lat, lon], radius=2000, color="crimson",
                    weight=2, fill=False, dash_array="5, 5"
                ).add_to(m)

                # Маркер центра клиники
                folium.Marker(
                    [lat, lon], tooltip="Центр анализа",
                    icon=folium.Icon(color="red", icon="plus", prefix="fa")
                ).add_to(m)

                # Фильтруем значения > 0 для текущего выбранного слоя
                active_vals = [met[filter_key] for met in hex_metrics.values() if met[filter_key] > 0]
                
                if not active_vals:
                    st.info("В выбранном радиусе нет объектов для отображения этого слоя. Попробуйте сменить слой.")
                else:
                    vmin, vmax = min(active_vals), max(active_vals)
                    if vmax == vmin:
                        vmax = vmin + 1

                    palettes = {
                        "population": ["#ffffcc", "#ffeda0", "#fed976", "#feb24c", "#fd8d3c", "#fc4e2a", "#e31a1c", "#bd0026"],
                        "floors":     ["#f7f4f9", "#e7e1ef", "#d4b9da", "#c994c7", "#df65b0", "#e7298a", "#ce1256", "#91003f"],
                        "traffic":    ["#313695", "#4575b4", "#74add1", "#abd9e9", "#fee090", "#fc8d59", "#d73027", "#a50026"],
                        "income":     ["#ffffe5", "#f7fcb9", "#d9f0a3", "#addd8e", "#78c679", "#41ab5d", "#238443", "#004529"],
                        "competition":["#1a9850", "#66bd63", "#a6d96a", "#d9ef8b", "#fee08b", "#fdae61", "#f46d43", "#d73027"],
                    }
                    
                # Словарь текстовых шаблонов для всплывающих подсказок
                tooltips_map = {
                    "population": "Примерное население ячейки: {} чел.",
                    "floors": "Средняя этажность зоны: {} эт.",
                    "traffic": "Дорожных путей в ячейке: {} ед.",
                    "income": "Бизнес-объектов в ячейке: {} ед.",
                    "competition": "Медицинских конкурентов: {} шт."
                }

                # Наносим гексагоны на карту динамически
                for h in grid:
                    val = hex_metrics[h][filter_key]
                    
                    # Если в выбранном слое значение 0 — ячейка полностью скрывается (как в Колабе)
                    if val == 0:
                        continue
                        
                    color = _hex_color(val, vmin, vmax, palettes[filter_key])
                    
                    try:
                        boundary = h3.cell_to_boundary(h)
                    except AttributeError:
                        boundary = h3.h3_to_geo_boundary(h, geo_json=False)
                        
                    coords = [(lat_pt, lon_pt) for lat_pt, lon_pt in boundary]
                    
                    if coords:
                        folium.Polygon(
                            locations=coords,
                            color="gray",
                            weight=0.3,
                            fill_color=color,
                            fill_opacity=0.6,
                            tooltip=tooltips_map[filter_key].format(val)
                        ).add_to(m)

                # Выводим только карту, таблица лидеров удалена
                st_folium(
                    m, 
                    width="100%", 
                    height=600, 
                    returned_objects=[], 
                    key="map_overpass_v9_fixed"
                )

        except Exception as e:
            st.error(f"Ошибка выполнения гео-анализа: {e}")
else:
    st.info("Запустите анализ локации, чтобы построить карту с гексами.")



# ==============================================================================
# СБРОС КЛЮЧА
# ==============================================================================

with st.sidebar:
    st.header("Сессия")

    st.success("OpenAI API-ключ активен для текущей сессии.")

    if st.button("Сбросить OpenAI ключ"):
        st.session_state.clear()
        st.cache_data.clear()
        st.rerun()

    st.caption(
        "После изменения портрета пациента или адреса просто нажмите "
        "«Запустить анализ». Ключ повторно вводить не нужно."
    )
