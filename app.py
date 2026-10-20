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
        + list(st.session_state.get("hex_chain") or [])))
    _name = "Без названия"  # имя объекта в GeoJSON (поле ввода убрано — не требовалось)

    def _poly_rings_geojson(p):
        # кольца полигона в порядке GeoJSON: сначала внешнее, затем дырки;
        # shapely-координаты уже замкнуты (первая = последней)
        rings = [[[lng, lat] for lng, lat in p.exterior.coords]]
        rings += [[[lng, lat] for lng, lat in h.coords] for h in p.interiors]
        return rings

    # стиль объекта — Яндекс Map Constructor читает эти properties при
    # импорте (соответствуют полям «Описание / Контур / Прозрачность /
    # Толщина / Заливка» в редакторе объекта)
    _style_props = {
        "description": (f"Выделено {len(_cells)} гексов "
                        f"(H3 res {_res0}). GeoHex Analytics"),
        "stroke": "#1f6fd6",       # цвет контура
        "stroke-width": 3,         # толщина контура, px
        # NB: Яндекс кладёт эти числа в поле «Прозрачность» КАК ЕСТЬ —
        # указываем именно прозрачность: 0.4 -> обводка на 60% непрозрачна,
        # 0.2 -> заливка на 80% непрозрачна
        "stroke-opacity": 0.4,
        "fill": "#1f6fd6",         # цвет заливки
        "fill-opacity": 0.2,
    }
    _geom = _dissolve_hexes(_cells)   # общие границы растворены
    _features = []
    if _geom is not None:
        _polys = ([_geom] if _geom.geom_type == "Polygon"
                  else list(_geom.geoms))
        for _fid, _p in enumerate(_polys):
            _p = _shapely_orient(_p, sign=1.0)  # внешнее кольцо против
            # часовой (RFC 7946) — иначе часть рендереров теряет фрагменты
            _features.append({"type": "Feature", "id": _fid,
                              "properties": dict(_style_props),
                              "geometry": {"type": "Polygon",
                                           "coordinates": _poly_rings_geojson(_p)}})

    _export_gj = {"type": "FeatureCollection",
                  "metadata": {"name": _name,
                               "creator": "Yandex Map Constructor"},
                  "features": _features}
    st.download_button(
        f"⬇ Скачать GeoJSON выделенных гексов ({len(_cells)})",
        data=json.dumps(_export_gj, ensure_ascii=False),
        file_name="hexes.geojson",
        mime="application/geo+json",
        help="Импортируется в Яндекс Карты: «Мои карты» → создание карты → "
             "импорт GeoJSON. Координаты [lon, lat], кольца замкнуты.")

_src = None
if map_type.startswith("1."):
    _src = ("Kontur (2023)" if kontur_df is not None
            else "OSM: суррогат")

render_map(grid, series, unit, geo, map_type, marker=marker,
           points=points, hex_extra=hex_extra, extra_aliases=extra_aliases,
           legend=legend, source=_src,
           yzoom={7: 13, 8: 15, 9: 16, 10: 17}.get(res_eff, 15),
           # ══ ГАММА РАСКРАСКИ ШКАЛЫ (0.75) ══════════════════════════════
           # Показатель нелинейности раскраски гексов: цвет считается как
           # t = (v / vmax) ** gamma, где v — значение гекса, vmax — максимум
           # по городу, t — позиция на палитре COLORS (0 = синий, 1 = красный).
           #   < 1 (напр. 0.4): жёлто-оранжевые начинаются при меньших
           #     значениях, но разница между близкими ВЫСОКИМИ значениями
           #     стирается (у Kontur Population масса гексов в верхней
           #     половине диапазона — там нужна гамма ближе к 1 или выше).
           #   > 1: верх диапазона растянут, разница высоких значений
           #     хорошо различима, но низ дольше остаётся сине-зелёным.
           # Подбирается на глаз под конкретный датасет. Слайдер убран из UI —
           # менять значение только здесь. Легенда под шкалой всегда считается
           # с той же гаммой, расхождения шкалы и заливки не будет.
           gamma=0.75,
           circ_center=_circ_center, center_is_marker=_use_marker)
if map_type.startswith("1.") and kontur_df is None:
    st.caption("⚠️ Оценки по OSM-зданиям — суррогатные: не учитывают реальное "
               "заселение и незавершённое строительство. Для точной численности "
               "загрузите Kontur Population (data.humdata.org, датасет "
               "«Population Density for 400m H3 Hexagons»).")
