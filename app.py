                else st.session_state.get("circle_center"))

if map_type.startswith("3."):
    st.info("**Из чего складывается индекс спроса.** Каждая строка тултипа — "
            "отдельный показатель, нормированный в индекс 0–100 относительно "
            "максимума по городу: «Жилой фонд: 25» значит, что объём жилья в этом "
            "гексе равен 25% от объёма в самом плотном гексе города. "
            "**Жилой фонд** — м² жилых зданий (площадь × этажность). **Розница/общепит** — "
            "непродовольственные магазины, кафе и рестораны. **Банки/офисы** — банки, "
            "обменники, офисы. **Остановки** — остановки ОТ. **Парковки/АЗС** — парковки "
            "и заправки. Итоговый индекс = 0.40·ЖилойФонд + 0.20·Розница + "
            "0.15·Банки + 0.15·Остановки + 0.10·Авто.")

if map_type.startswith("6."):
    st.info("**Как считается индекс трафика.** Берётся суммарная длина дорог нужного "
            "класса внутри гекса, делится на площадь гекса (км/км²), затем нормируется "
            "в индекс 0–100 относительно самого загруженного гекса города. "
            "**Пешеходный**: footway, pedestrian, path, steps, cycleway, living_street. "
            "**Автомобильный**: motorway–tertiary и съезды. Тултип каждого гекса показывает "
            "оба индекса сразу. Это интенсивность сети, а не реальные потоки машин — "
            "для потоков нужны платные данные (TomTom и т.п.).")

c1, c2, c3 = st.columns(3)
c1.metric("Гексов в сетке", f"{len(grid):,}")
c2.metric("Resolution", f"res {res_eff} (~{RES_SPACING_KM[res_eff]} км между центрами)")
c3.metric("Максимум в ячейке", f"{series.max():,.0f} {unit}" if len(series) else "—")

legend = None
if map_type.startswith("4."):
    _sel = [c for c in POI_CATEGORIES if not sub_option or c in sub_option]
    legend = [(name, color) for name, (color, _) in POI_CATEGORIES.items()
              if name in _sel]
elif map_type.startswith("5."):
    legend = [(name, color) for name, (color, _) in SOCIAL_GROUPS.items()
              if sub_option is None or name in sub_option]
elif map_type.startswith("7."):
    legend = [(name, color) for name, (color, _) in COMPETITOR_TYPES.items()
              if sub_option is None or name in sub_option]

extra_aliases = None
if map_type.startswith("3."):
    extra_aliases = {"c_people": "Жилой фонд: ", "c_retail": "Розница/общепит: ",
                     "c_biz": "Банки/офисы: ", "c_transit": "Остановки: ",
                     "c_auto": "Парковки/АЗС: "}
elif map_type.startswith("4."):
    _sel = [c for c in POI_CATEGORIES if not sub_option or c in sub_option]
    extra_aliases = {name: f"{name}: " for name in POI_CATEGORIES if name in _sel}
elif map_type.startswith("6."):
    extra_aliases = {"ped": "Пешеходный трафик: ", "auto": "Автомобильный трафик: "}
elif map_type.startswith("5."):
    extra_aliases = {name: f"{name}: " for name in SOCIAL_GROUPS
                     if sub_option is None or name in sub_option}
elif map_type.startswith("7."):
    extra_aliases = {name: f"{name}: " for name in COMPETITOR_TYPES
                     if sub_option is None or name in sub_option}


# ---- сумма показателей внутри кругов: НЕ по умолчанию, а по кнопке.
# Карты 1–7: население / жилфонд / индекс спроса / объекты (с учётом
# фильтров) / социнфраструктура / индекс трафика / мед. объекты. Серия
# показателей уже отражает текущие фильтры и режим. Гексы берутся по
# центрам внутри радиуса. Привязка результата: тип карты (очистка выше)
# + гекс + отпечаток фильтров (при смене фильтров сумма скрывается).
if map_type.startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.")):
    _sums_center = _circ_center
    # отпечаток текущих фильтров/настроек, влияющих на серию показателей
    _fkey = f"{sub_option}|{'K' if kontur_df is not None else ''}|{m2_per_person}"
    if st.button("Σ  Сумма показателей в кругах", disabled=not _sums_center,
                 help="По гексам текущей карты (с учётом фильтров), центры "
                      "которых попадают в радиус 1, 2 и 5 км: сумма для "
                      "карт 1, 2, 4, 5, 7 и СРЕДНЕЕ для индексных карт 3 и 6. "
                      "После смены фильтров нажмите кнопку ещё раз."):
        _s = sums_in_circles(series, _sums_center)
        if _s is not None:
            st.session_state["circle_sums"] = {"sums": _s, "unit": unit,
                                               "center": _sums_center,
                                               "map_type": map_type,
                                               "fkey": _fkey}
        else:
            st.session_state.pop("circle_sums", None)
            st.warning("Нет данных для суммирования на текущей карте.")

    # показываем результат, только пока он относится к ТЕКУЩЕМУ выбранному
    # гексу: перенёс круги — сумма скроется, пока не нажата кнопка снова
    _saved_sums = st.session_state.get("circle_sums")
    if (_saved_sums
            and _saved_sums.get("center") == _circ_center
            and _saved_sums.get("fkey") == _fkey):  # фильтры не менялись
        _s1, _n1 = _saved_sums["sums"][1.0]
        _s2, _n2 = _saved_sums["sums"][2.0]
        _s5, _n5 = _saved_sums["sums"][5.0]
        # для индексных карт (3 — спрос, 6 — трафик) показываем СРЕДНЕЕ
        # по гексам радиуса: сумма индексов не имеет смысла
        _avg = map_type.startswith(("3.", "6."))
        _sum_col, _ = st.columns([1, 2])  # узкая колонка слева, метрики в столбик

        def _r_metric(_r_label, _s, _n):
            if _avg:
                _v = _s / _n if _n else 0.0
                st.metric(f"Среднее в радиусе {_r_label} · {_n} гексов",
                          f"{_v:,.1f} {_saved_sums['unit']}".rstrip())
            else:
                st.metric(f"Σ в радиусе {_r_label} · {_n} гексов",
                          f"{_s:,.0f} {_saved_sums['unit']}".rstrip())

        with _sum_col:
            _r_metric("1 км", _s1, _n1)
            _r_metric("2 км", _s2, _n2)
            _r_metric("5 км", _s5, _n5)
    if map_type.startswith(("4.", "5.", "7.")):
        st.caption("💡 Сумма считается по ТЕКУЩИМ фильтрам. Изменили "
                   "категории — нажмите кнопку ещё раз, чтобы пересчитать.")

# ---- сводка по ВСЕМ картам в Excel: одна кнопка вместо обхода 7 карт.
# Считает все показатели с ПОЛНЫМИ фильтрами для текущего центра кругов
# (метка или гекс) и радиусов 1/2/5 км.
if _circ_center:
    # ОДНА кнопка на фичу: пока сводка не готова для текущего центра и
    # разрешений — «Сформировать», после расчёта она заменяется «Скачать».
    # Всегда включает комплект для res 8, плюс выбранный res, если он не 8.
    _res_list = list(dict.fromkeys([8, res_eff]))
    _exp = st.session_state.get("allmaps_export")
    _fresh = (_exp and _exp.get("center") == _circ_center
              and _exp.get("res_list") == _res_list)
    if not _fresh:
        if st.button("📊 Сформировать сводку по всем картам (Excel)",
                     help="Считает показатели всех 7 карт (трафик — пеший и "
                          "авто отдельными строками) с полными фильтрами для "
                          "текущего центра кругов и радиусов 1/2/5 км. Всегда "
                          "включает комплект для res 8; если выбрано другое "
                          "разрешение — добавит второй комплект листов. "
                          "Первый запуск 10–30 сек, дальше — из кэша."):
            with st.spinner("Считаю все 7 карт для сводки (первый раз 10–30 сек)…"):
                try:
                    _results = {_r: build_all_maps_summary(
                        _circ_center, _r, stored, kontur_df, m2_per_person)
                        for _r in _res_list}
                    _buf = io.BytesIO()
                    _fmt = "xlsx"
                    try:
                        with pd.ExcelWriter(_buf, engine="openpyxl") as _w:
                            if len(_res_list) == 1:
                                _sum_df, _par_df = _results[_res_list[0]]
                                _par_df.to_excel(_w, sheet_name="Параметры",
                                                 index=False)
                                _sum_df.to_excel(_w, sheet_name="Сводка",
                                                 index=False)
                            else:
                                for _r in _res_list:
                                    _sum_df, _par_df = _results[_r]
                                    _par_df.to_excel(
                                        _w, sheet_name=f"Параметры res {_r}",
                                        index=False)
                                    _sum_df.to_excel(
                                        _w, sheet_name=f"Сводка res {_r}",
                                        index=False)
                        _mime = ("application/vnd.openxmlformats-"
                                 "officedocument.spreadsheetml.sheet")
                    except ImportError:
                        # openpyxl нет — один CSV с колонкой Resolution
                        _frames = []
                        for _r in _res_list:
                            _s = _results[_r][0].copy()
                            _s.insert(0, "Resolution", f"res {_r}")
                            _frames.append(_s)
                        _buf = io.BytesIO(pd.concat(_frames).to_csv(index=False)
                                          .encode("utf-8-sig"))
                        _fmt, _mime = "csv", "text/csv"
                        st.warning("⚠️ openpyxl не установлен — отдаю CSV. "
                                   "Для xlsx добавьте openpyxl>=3.1 в "
                                   "requirements.txt")
                    st.session_state["allmaps_export"] = {
                        "data": _buf.getvalue(), "ext": _fmt, "mime": _mime,
                        "center": _circ_center, "res_list": _res_list}
                except Exception as e:  # noqa: BLE001 — любая ошибка расчёта
                    st.session_state.pop("allmaps_export", None)
                    st.error(f"Не удалось сформировать сводку: {e}")
    else:
        st.download_button(f"⬇ Скачать сводку по всем картам ({_exp['ext']})",
                           data=_exp["data"],
                           file_name=f"geohex_svodka.{_exp['ext']}",
                           mime=_exp["mime"])

# ---- экспорт выделенных гексов в GeoJSON (формат Yandex Map Constructor):
# FeatureCollection + metadata{name, creator} + features c Polygon-кольцами
# в порядке [lon, lat], кольцо замкнуто. Работает в обоих режимах: в
# одиночном экспортируется выбранный гекс, в мультивыборе — вся цепочка.
if st.session_state.get("circle_center"):
    _res0 = h3.get_resolution(grid[0])
    _cells = list(dict.fromkeys(
        [h3.latlng_to_cell(st.session_state["circle_center"][0],
                           st.session_state["circle_center"][1], _res0)]
