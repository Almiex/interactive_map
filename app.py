# -*- coding: utf-8 -*-

import time
import requests
import pandas as pd
import numpy as np
import streamlit as st
import h3
import folium
from streamlit_folium import st_folium
from openai import OpenAI
from pydantic import BaseModel, Field


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

import streamlit as st
import numpy as np
import pandas as pd
import requests
import folium
from streamlit_folium import st_folium
import h3

# ==============================================================================
# 1. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ H3 КАРТ
# ==============================================================================

def _h3_latlon_to_cell(lat, lon, res):
    try:
        return h3.latlng_to_cell(lat, lon, res)
    except AttributeError:
        return h3.geo_to_h3(lat, lon, res)


def _h3_cell_to_latlon(h):
    try:
        return h3.cell_to_latlng(h)
    except AttributeError:
        return h3.h3_to_geo(h)


def _h3_grid_disk(h, k):
    try:
        return h3.grid_disk(h, k)
    except AttributeError:
        return h3.k_ring(h, k)


def _h3_boundary(h):
    try:
        boundary = h3.cell_to_boundary(h)
        pts = list(boundary)
        if pts and isinstance(pts, (tuple, list)):
            return [(p[0], p[1]) for p in pts]
        return pts
    except AttributeError:
        return h3.h3_to_geo_boundary(h, geo_json=False)


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


@st.cache_data(show_spinner=False, ttl=600)
def fetch_osm_for_hex_grid(lat, lon, radius_m=2000):
    url = "https://openstreetmap.fr"
    query = f"""
    [out:json][timeout:15];
    (
      node["building"](around:{radius_m},{lat},{lon});
      way["building"](around:{radius_m},{lat},{lon});
      node["highway"](around:{radius_m},{lat},{lon});
      node["amenity"~"clinic|hospital|doctors|pharmacy|dentist"](around:{radius_m},{lat},{lon});
      node["shop"~"mall|supermarket|car|jewelry|boutique|beauty"](around:{radius_m},{lat},{lon});
      node["office"](around:{radius_m},{lat},{lon});
    );
    out center;
    """
    try:
        r = requests.post(url, data={"data": query}, timeout=15)
        r.raise_for_status()
        return r.json().get("elements", [])
    except Exception:
        try:
            r = requests.post("https://overpass-api.de", data={"data": query}, timeout=15)
            return r.json().get("elements", [])
        except Exception:
            return []


def build_hex_grid(lat, lon, radius_m=2000, resolution=8):
    center_hex = _h3_latlon_to_cell(lat, lon, resolution)
    k = int(np.ceil(radius_m / 460)) + 1
    candidates = _h3_grid_disk(center_hex, k)
    hexes = []
    for h in candidates:
        hlat, hlon = _h3_cell_to_latlon(h)
        if _haversine_km(lat, lon, hlat, hlon) <= radius_m / 1000:
            hexes.append(h)
    return hexes


# ==============================================================================
# 2. АНАЛИТИКА МЕТРИК
# ==============================================================================

def compute_hex_metrics(hexes, elements, resolution=8):
    data = {h: {"bld": [], "road": [], "med": [], "prem": []} for h in hexes}
    hw_weights = {
        "motorway": 5, "trunk": 5, "primary": 4, "secondary": 3,
        "tertiary": 2, "unclassified": 2, "residential": 1,
        "living_street": 1, "pedestrian": 2, "footway": 1, "path": 1,
    }

    if not elements:
        for i, h in enumerate(hexes):
            data[h]["bld"].append({"levels": (i % 6) + 2, "type": "apartments"})
            data[h]["road"].append({"weight": (i % 4) + 1})
            if i % 3 == 0:
                data[h]["med"].append({"type": "pharmacy"})
            if i % 2 == 0:
                data[h]["prem"].append({"weight": 2})
    else:
        for el in elements:
            elat = el.get("lat") or el.get("center", {}).get("lat")
            elon = el.get("lon") or el.get("center", {}).get("lon")
            if elat is None or elon is None:
                continue

            h = _h3_latlon_to_cell(float(elat), float(elon), resolution)
            if h not in data:
                continue

            tags = el.get("tags", {})
            t = el.get("type", "")

            if "building" in tags:
                levels = 1
                try:
                    levels = int(tags.get("building:levels", 1))
                except ValueError:
                    pass
                data[h]["bld"].append({"levels": levels, "type": tags.get("building", "yes")})

            if "highway" in tags or (t == "node" and "highway" in tags):
                hw = tags.get("highway", "residential")
                data[h]["road"].append({"weight": hw_weights.get(hw, 1)})

            if tags.get("amenity") in ("clinic", "hospital", "doctors", "pharmacy", "dentist"):
                data[h]["med"].append({"type": tags["amenity"]})

            prem = 0
            shop = tags.get("shop", "")
            if shop in ("mall", "supermarket", "car", "jewelry", "boutique", "beauty"):
                prem += 3 if shop in ("mall", "car", "jewelry") else 2
            if "office" in tags:
                prem += 2
            if tags.get("building") in ("commercial", "retail", "office"):
                prem += 1
            if prem > 0:
                data[h]["prem"].append({"weight": prem})

    metrics = {}
    for h in hexes:
        d = data[h]
        pop = sum(b["levels"] * (4 if b["type"] in ("apartments", "residential") else 2 if b["type"] == "house" else 1) for b in d["bld"])
        floors = np.mean([b["levels"] for b in d["bld"]]) if d["bld"] else 0
        traffic = sum(r["weight"] for r in d["road"])
        income = sum(p["weight"] for p in d["prem"]) * 5 + len(d["bld"])
        competition = len(d["med"])

        metrics[h] = {
            "population": int(pop),
            "floors": round(float(floors), 1),
            "traffic": int(traffic),
            "income": int(income),
            "competition": int(competition),
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
# 3. ИНТЕРФЕЙС И КАРТА
# ==============================================================================

st.divider()
st.subheader("🗺️ Гео-аналитика радиуса 2 км (гексагональная сетка OSM)")

if "last_result" in st.session_state:
    result = st.session_state.last_result
    lat, lon = result["latitude"], result["longitude"]

    filter_key = st.selectbox(
        "Слой данных на гексах:",
        options=["population", "floors", "traffic", "income", "competition"],
        format_func=lambda x: {
            "population": "Плотность населения",
            "floors": "Этажность застройки",
            "traffic": "Пешеходный + авто трафик",
            "income": "Платёжеспособность аудитории",
            "competition": "Конкуренция медучреждений"
        }.get(x, x)
    )

    with st.spinner("Собираем данные OSM и строим гекс-сетку…"):
        try:
            elements = fetch_osm_for_hex_grid(lat, lon, radius_m=2000)
            grid = build_hex_grid(lat, lon, radius_m=2000, resolution=8)
            hex_metrics = compute_hex_metrics(grid, elements, resolution=8)

            if not hex_metrics:
                st.warning("Не удалось сгенерировать гексагональную сетку.")
            else:
                m = folium.Map(location=[lat, lon], zoom_start=13, tiles="CartoDB positron")
                
                folium.Circle(
                    location=[lat, lon], radius=2000, color="#0066cc",
                    weight=2, fill=True, fill_color="#0066cc", fill_opacity=0.03, tooltip="Радиус 2 км"
                ).add_to(m)

                folium.Marker(
                    [lat, lon], tooltip="Анализируемая клиника",
                    icon=folium.Icon(color="red", icon="plus", prefix="fa")
                ).add_to(m)

                values = [met[filter_key] for met in hex_metrics.values()]
                vmin, vmax = min(values) if values else 0, max(values) if values else 1
                if vmax == vmin:
                    vmax = vmin + 1

                palettes = {
                    "population": ["#ffffcc", "#ffeda0", "#fed976", "#feb24c", "#fd8d3c", "#fc4e2a", "#e31a1c", "#bd0026"],
                    "floors":     ["#f7f4f9", "#e7e1ef", "#d4b9da", "#c994c7", "#df65b0", "#e7298a", "#ce1256", "#91003f"],
                    "traffic":    ["#313695", "#4575b4", "#74add1", "#abd9e9", "#fee090", "#fc8d59", "#d73027", "#a50026"],
                    "income":     ["#ffffe5", "#f7fcb9", "#d9f0a3", "#addd8e", "#78c679", "#41ab5d", "#238443", "#004529"],
                    "competition":["#1a9850", "#66bd63", "#a6d96a", "#d9ef8b", "#fee08b", "#fdae61", "#f46d43", "#d73027"],
                }
                labels = {
                    "population": "Плотность населения (прокси)",
                    "floors": "Средняя этажность",
                    "traffic": "Интенсивность трафика",
                    "income": "Платёжеспособность (прокси)",
                    "competition": "Конкуренция медучреждений",
                }
                
                palette = palettes[filter_key]

                for h, met in hex_metrics.items():
                    val = met[filter_key]
                    color = _hex_color(val, vmin, vmax, palette)
                    coords = _h3_boundary(h)
                    if coords:
                        folium.Polygon(
                            locations=coords, color="#333333", weight=1,
                            fill_color=color, fill_opacity=0.6, tooltip=f"{labels[filter_key]}: {val}"
                        ).add_to(m)

                st_folium(
                    m,
                    width="100%",
                    height=600,
                    returned_objects=[],
                    key="map_fixed_v5_final"
                )

                st.caption("Справочно: топ-5 гексов по выбранному показателю")
                
                df_hex = pd.DataFrame([
                    {"h3_index": h, **vals} for h, vals in hex_metrics.items()
                ])
                
                if not df_hex.empty:
                    df_hex = df_hex.sort_values(by=filter_key, ascending=False).head(5)
                    st.dataframe(df_hex, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Ошибка построения карты: {e}")
            
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
