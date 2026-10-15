# -*- coding: utf-8 -*-
"""
GeoHex Analytics — Streamlit-сервис аналитики города по гексагональной сетке Uber H3.

Источники (все бесплатные):
  * Геокодинг городов        — Nominatim (OSM)
  * Здания / дороги / POI    — Overpass API (OpenStreetMap)
  * Реальная численность     — Kontur Population (локальный файл, https://data.humdata.org)

Запуск:  streamlit run app.py
"""

import io
import re
import json
import time
import math
import logging
import hashlib
import gc
import requests
import concurrent.futures

logger = logging.getLogger("geohex")  # логи видны в Streamlit Cloud: Manage app → Logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    force=True)
import numpy as np
import pandas as pd
import h3
import folium
import streamlit as st
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import unary_union, orient as _shapely_orient
from streamlit_folium import st_folium
from branca.colormap import LinearColormap
from jinja2 import Template as _Jinja2Template

# --------------------------------------------------------------------------- #
#  Константы
# --------------------------------------------------------------------------- #
# Порядок — по наблюдаемой живучести с нашего деплоя.
# overpass-api.de УБРАН: со Streamlit Cloud он стабильно отвечает
# Connection refused (TCP-бан IP), попытки только съедали время.
OVERPASS_ENDPOINTS = [
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.nchc.org.tw/api/interpreter",
    "https://overpass.maps.mail.ru/api/interpreter",
]
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
# резервный геокодер (komoot) — подхватывает город, пока Nominatim режет
# запросы с нашего IP (на shared-хостинге лимит часто срабатывает из-за
# чужого трафика). Полигона границы не отдаёт — сетка построится по bbox
PHOTON_URL = "https://photon.komoot.io/api/"
HEADERS = {"User-Agent": "GeoHexAnalytics/1.0 (educational; OSM data)"}

# ── Защита от бана Nominatim ──────────────────────────────────────────────
# Публичный Nominatim разрешает ~1 запрос/сек с IP; на shared-хостинге IP
# общий с чужими приложениями — лимит срабатывает «сам». Поэтому:
# 1) throttle: приложение физически не шлёт запросы чаще интервала;
# 2) без ретраев при 429: повтор на общем IP почти всегда снова 429,
#    а агрессивные попытки продлевают блокировку. При 429 уходим на резерв.
NOMINATIM_MIN_INTERVAL = 1.2    # минимальный интервал между запросами, сек
_last_nominatim_ts = [0.0]      # изменяемая ячейка: время последнего запроса


def _nominatim_throttle():
    """Гарантирует NOMINATIM_MIN_INTERVAL между запросами к Nominatim
    по ВСЕМУ приложению (город, адрес, догеокодинг за полигоном)."""
    now = time.monotonic()
    gap = now - _last_nominatim_ts[0]
    if gap < NOMINATIM_MIN_INTERVAL:
        time.sleep(NOMINATIM_MIN_INTERVAL - gap)
    _last_nominatim_ts[0] = time.monotonic()

# расстояние между центрами соседних гексов, км (по стандарту Uber H3)
RES_SPACING_KM = {7: 2.4, 8: 0.92, 9: 0.35, 10: 0.13}

MAX_GRID_CELLS = 50000  # потолок сетки; canvas-рендер держит десятки тысяч полигонов

MAP_TYPES = [
    "1. Плотность населения",
    "2. Объём жилого фонда",
    "3. Индекс спроса (платёжеспособность)",
    "4. Инфраструктура и POI",
    "5. Социальная инфраструктура",
    "6. Трафик (индекс, пеший/авто)",
    "7. Мед. учреждения и аптеки",
]

# ПВЗ в OSM не имеют единого тега — находим по названию/бренду
PVZ_RE = re.compile(
    r"ozon|озон|wildberries|вайлдберриз|сдэк|cdek|авито|avito|"
    r"яндекс[ -]?маркет|yandex[ -]?market", re.IGNORECASE)
PVZ_WB_RE = re.compile(r"(^|[^a-z])wb([^a-z]|$)|(^|\s)вб(\s|$)", re.IGNORECASE)


def is_pvz(tags: dict) -> bool:
    text = f"{tags.get('name', '')} {tags.get('brand', '')}"
    return bool(PVZ_RE.search(text) or PVZ_WB_RE.search(text))


def _name_of(tags: dict) -> str:
    return f"{tags.get('name', '')} {tags.get('brand', '')}".lower()


# категория -> (цвет точки, matcher по тегам)
POI_CATEGORIES = {
    "Торговые центры": ("#e377c2",
                        lambda t: t.get("shop") in {"mall", "department_store"}
                        or re.search(r"торгов(ый|ого) центр|(^|\s)тц(\s|$)|молл", _name_of(t))),
    "Бизнес-центры и офисы": ("#17becf",
                              lambda t: "office" in t
                              or re.search(r"бизнес[ -]?центр|business center|(^|\s)бц(\s|$)",
                                           _name_of(t))),
    "ПВЗ маркетплейсов (Ozon, WB, СДЭК, Авито, Яндекс)": ("#d62728", is_pvz),
    "Продуктовые магазины": ("#2ca02c",
                             lambda t: t.get("shop") in
                             {"supermarket", "convenience", "greengrocer", "deli",
                              "butcher", "bakery", "seafood"}),
    "Общепит": ("#ff7f0e",
                lambda t: t.get("amenity") in
                {"restaurant", "cafe", "fast_food", "bar", "pub",
                 "food_court", "biergarten"}),
    "Аптеки": ("#1f6fd6", lambda t: t.get("amenity") == "pharmacy"),
