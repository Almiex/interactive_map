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

import streamlit as st
import numpy as np
import pandas as pd
import requests
import folium
from streamlit_folium import st_folium
import h3

def get_clean_hex_analytics(center_lat, center_lng, radius_meter=2000, h3_resolution=9):
    """
    Скачивает реальные слои из OSMnx, рассчитывает население по площади (из Колаба)
    и собирает честные физические метрики для каждого гексагона.
    """
    # 1. Сбор жилых зданий для населения и этажности
    building_tags = {"building": ["apartments", "residential", "house", "living_quarter"]}
    try:
        gdf_buildings = ox.features_from_point((center_lat, center_lng), tags=building_tags, dist=radius_meter)
    except Exception:
        gdf_buildings = gpd.GeoDataFrame()

    population_records = []
    floor_records = []

    if not gdf_buildings.empty:
        # Переводим в метры для точного расчета площади, как в вашем Колабе!
        gdf_buildings_meters = gdf_buildings.to_crs(epsg=3857)

        for idx, row in gdf_buildings.iterrows():
            try:
                footprint_area = gdf_buildings_meters.loc[idx].geometry.area
                
                levels = row.get('building:levels', None)
                if pd.isna(levels) or not str(levels).isdigit():
                    b_type = row.get('building', 'residential')
                    levels = 9 if b_type == 'apartments' else 5
                else:
                    levels = int(levels)

                # Ваша оригинальная формула из Колаба
                total_living_area = footprint_area * levels
                estimated_people = int(total_living_area / 27)

                if estimated_people < 1:
                    estimated_people = 2

                # Находим центр здания для привязки к H3
                centroid = row.geometry.centroid
                hex_id = h3.latlng_to_cell(centroid.y, centroid.x, h3_resolution)

                population_records.append({"hex_id": hex_id, "people": estimated_people})
                floor_records.append({"hex_id": hex_id, "levels": levels})
            except Exception:
                continue

    # 2. Сбор инфраструктурных слоев (дороги, бизнес, медицина)
    infra_tags = {
        "highway": ["motorway", "trunk", "primary", "secondary", "tertiary", "residential", "living_street", "pedestrian", "footway"],
        "amenity": ["clinic", "hospital", "doctors", "pharmacy", "dentist"],
        "shop": ["mall", "supermarket", "car", "jewelry", "boutique", "beauty"],
        "office": True
    }
    try:
        gdf_infra = ox.features_from_point((center_lat, center_lng), tags=infra_tags, dist=radius_meter)
    except Exception:
        gdf_infra = gpd.GeoDataFrame()

    traffic_records = []
    business_records = []
    med_records = []

    if not gdf_infra.empty:
        for idx, row in gdf_infra.iterrows():
            try:
                centroid = row.geometry.centroid
                hex_id = h3.latlng_to_cell(centroid.y, centroid.x, h3_resolution)
                
                # Фильтруем по типу инфраструктуры
                if hasattr(row, 'highway') and pd.notna(row.get('highway')):
                    traffic_records.append({"hex_id": hex_id, "count": 1})
                if hasattr(row, 'amenity') and row.get('amenity') in ["clinic", "hospital", "doctors", "pharmacy", "dentist"]:
                    med_records.append({"hex_id": hex_id, "count": 1})
                if (hasattr(row, 'shop') and pd.notna(row.get('shop'))) or (hasattr(row, 'office') and pd.notna(row.get('office'))):
                    business_records.append({"hex_id": hex_id, "count": 1})
            except Exception:
                continue

    # 3. Генерируем базовую сетку гексагонов вокруг центра
    center_hex = h3.latlng_to_cell(center_lat, center_lng, h3_resolution)
    max_rings = int(radius_meter / 180) + 1
    all_hexes = h3.grid_disk(center_hex, max_rings)

    # Агрегируем все списки в единые словари по hex_id
    df_p = pd.DataFrame(population_records)
    pop_map = df_p.groupby("hex_id")["people"].sum().to_dict() if not df_p.empty else {}

    df_f = pd.DataFrame(floor_records)
    floor_map = df_f.groupby("hex_id")["levels"].mean().to_dict() if not df_f.empty else {}

    df_t = pd.DataFrame(traffic_records)
    traffic_map = df_t.groupby("hex_id")["count"].sum().to_dict() if not df_t.empty else {}

    df_b = pd.DataFrame(business_records)
    biz_map = df_b.groupby("hex_id")["count"].sum().to_dict() if not df_b.empty else {}

    df_m = pd.DataFrame(med_records)
    med_map = df_m.groupby("hex_id")["count"].sum().to_dict() if not df_m.empty else {}

    # Собираем итоговую структуру метрик
    hex_metrics = {}
    for h in all_hexes:
        # Проверяем расстояние от центра, чтобы сетка была круглой (радиус 2км)
        h_lat, h_lon = h3.cell_to_latlng(h)
        
        # Считаем расстояние (упрощенно)
        R = 6371.0
        phi1, phi2 = np.radians(center_lat), np.radians(h_lat)
        dphi = np.radians(h_lat - center_lat)
        dlambda = np.radians(h_lon - center_lng)
        a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
        dist = 2 * R * np.arcsin(np.sqrt(a)) * 1000

        if dist <= radius_meter:
            hex_metrics[h] = {
                "population": int(pop_map.get(h, 0)),
                "floors": round(float(floor_map.get(h, 0.0)), 1),
                "traffic": int(traffic_map.get(h, 0)),
                "income": int(biz_map.get(h, 0)),
                "competition": int(med_map.get(h, 0)),
            }
            
    return hex_metrics

def _hex_color(value, vmin, vmax, palette):
    if vmax == vmin:
        return palette[0]
    ratio = (value - vmin) / (vmax - vmin)
    idx = int(ratio * (len(palette) - 1))
    idx = max(0, min(idx, len(palette) - 1))
    return palette[idx]


# ==============================================================================
# 3. ИНТЕРФЕЙС И ОТОБРАЖЕНИЕ КАРТЫ СЛОЕВ
# ==============================================================================

st.divider()
st.subheader("🗺️ Гео-аналитика радиуса 2 км (гексагональная сетка OSMnx)")

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

    with st.spinner("Загружаем геометрию OSMnx и рассчитываем метрики…"):
        try:
            # Вызываем нашу новую точную функцию
            hex_metrics = get_clean_hex_analytics(lat, lon, radius_meter=2000, h3_resolution=9)

            if not hex_metrics:
                st.warning("Не удалось собрать данные по указанной локации.")
            else:
                m = folium.Map(location=[lat, lon], zoom_start=13, tiles="CartoDB positron")
                
                # Радиус анализа
                folium.Circle(
                    location=[lat, lon], radius=2000, color="crimson",
                    weight=2, fill=False, dash_array="5, 5"
                ).add_to(m)

                # Маркер клиники
                folium.Marker(
                    [lat, lon], tooltip="Центр анализа",
                    icon=folium.Icon(color="red", icon="home")
                ).add_to(m)

                # Вычисляем vmin/vmax строго для активных (не нулевых) ячеек текущего слоя
                active_vals = [met[filter_key] for met in hex_metrics.values() if met[filter_key] > 0]
                
                if not active_vals:
                    st.info("В выбранном радиусе нет объектов для отображения этого слоя.")
                    vmin, vmax = 0, 1
                else:
                    vmin, vmax = min(active_vals), max(active_vals)
                    if vmax == vmin:
                        vmax = vmin + 1

                # Наборы палитр
                palettes = {
                    "population": ["#fcfbfd", "#efedf5", "#dadaeb", "#bcbddc", "#9e9ac8", "#807dba", "#6a51a3", "#4a1486"],
                    "floors":     ["#f7f4f9", "#e7e1ef", "#d4b9da", "#c994c7", "#df65b0", "#e7298a", "#ce1256", "#91003f"],
                    "traffic":    ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#084594"],
                    "income":     ["#f7fcf5", "#e5f5e0", "#c7e9b4", "#74c476", "#41ab5d", "#238443", "#005a32"],
                    "competition":["#fff5f0", "#fee0d2", "#fcbba1", "#fc9272", "#fb6a4a", "#ef3b2c", "#cb181d", "#990000"],
                }
                
                tooltips_map = {
                    "population": "Примерное население ячейки: {} чел.",
                    "floors": "Средняя этажность зоны: {} эт.",
                    "traffic": "Дорожная сеть в ячейке: {} ед.",
                    "income": "Бизнес-объекты (офисы/магазины): {} ед.",
                    "competition": "Медицинские учреждения: {} шт.",
                }

                # Наносим гексагоны на карту
                for h, met in hex_metrics.items():
                    val = met[filter_key]
                    
                    # ЖЕСТКАЯ ФИЛЬТРАЦИЯ: Если в этом слое у гексагона 0 — мы его просто не рисуем!
                    if val == 0:
                        continue
                        
                    color = _hex_color(val, vmin, vmax, palettes[filter_key])
                    
                    # Получаем границы гексагона
                    boundary = h3.cell_to_boundary(h)
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

                # Рендерим карту в Streamlit
                st_folium(m, width="100%", height=600, returned_objects=[], key="osm_clean_map_v9")

                # Обновление нижней таблицы лидеров
                st.caption("📊 Лидеры локации: топ-5 зон по выбранному показателю")
                table_rows = []
                for h, vals in hex_metrics.items():
                    if vals[filter_key] == 0:
                        continue
                    h_lat, h_lon = h3.cell_to_latlon(h)
                    table_rows.append({
                        "Координаты центра": f"{h_lat:.5f}, {h_lon:.5f}",
                        "population": vals["population"],
                        "floors": vals["floors"],
                        "traffic": vals["traffic"],
                        "income": vals["income"],
                        "competition": vals["competition"]
                    })
                
                df_hex = pd.DataFrame(table_rows)
                if not df_hex.empty:
                    df_hex = df_hex.sort_values(by=filter_key, ascending=False).head(5)
                    df_hex.insert(0, "Ранг зоны", [f"🏆 Локация #{i+1}" for i in range(len(df_hex))])
                    df_hex = df_hex.rename(columns={
                        "population": "Население (чел.)",
                        "floors": "Ср. этажность",
                        "traffic": "Дороги/пути (ед.)",
                        "income": "Бизнес (объекты)",
                        "competition": "Конкуренты (объекты)"
                    })
                    st.dataframe(df_hex, use_container_width=True, hide_index=True)

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
