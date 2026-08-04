import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

# Настройка страницы дашборда (желательно в самом верху)
st.set_page_config(page_title="Аналитический отчет клиники", layout="wide")

# Кастомные стили для таблиц и блоков, которые были в твоем Colab
st.markdown("""
    <style>
    .report-table { width: 100%; border-collapse: collapse; margin: 20px 0; font-family: sans-serif; }
    .report-table th { background-color: #005F73; color: white; padding: 10px; text-align: left; }
    .report-table td { padding: 8px; border-bottom: 1px solid #ddd; }
    .clinic-header { background-color: #f8f9fa; border-left: 5px solid #005F73; padding: 15px; margin-bottom: 25px; }
    .clinic-title { font-size: 24px; font-weight: bold; color: #2B2D42; }
    .clinic-subtitle { font-size: 14px; color: #6C9D9D; }
    </style>
""", unsafe_allow_html=True)

st.title("🏥 Аналитическая панель клиники")
st.write("Загрузите выгрузку из МИС в формате Excel для построения интерактивного отчета.")

# Шаг 1 & 2: Загрузка файла пользователем через интерфейс
uploaded_file = st.file_uploader("Выберите Excel файл (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    try:
        # Чтение листов
        df_clean = pd.read_excel(uploaded_file, sheet_name=0)
        df_meta = pd.read_excel(uploaded_file, sheet_name=1)
        
        # Сбор метаданных
        clinic_name = df_meta.iloc[0, 0] if not df_meta.empty else "ООО КЛИНИКА"
        period_str = df_meta.iloc[0, 1] if not df_meta.empty else "Период не указан"
        
        # Шаг 3: Расчет метрик и группировка
        # Предполагаем, что предобработка типов данных аналогична твоей
        sp_report = df_clean.groupby('Специализация', as_index=False).agg({
            'Табель': 'sum',
            'Занято записями': 'sum',
            'Дошло пациентов': 'sum'
        })
        
        sp_report['Свободно'] = sp_report['Табель'] - sp_report['Занято записями']
        sp_report['Потери'] = sp_report['Занято записями'] - sp_report['Дошло пациентов']
        
        sp_report['Загрузка %'] = np.where(sp_report['Табель'] > 0, (sp_report['Занято записями'] / sp_report['Табель']) * 100, np.nan)
        sp_report['Явка %'] = np.where(sp_report['Занято записями'] > 0, (sp_report['Дошло пациентов'] / sp_report['Занято записями']) * 100, np.nan)
        
        # Округляем для красоты
        sp_report = sp_report.round(1)

        # Вывод шапки клиники
        st.markdown(f"""
            <div class="clinic-header">
                <div class="clinic-title">🏥 КЛИНИКА: {clinic_name}</div>
                <div class="clinic-subtitle">📊 Аналитический отчет: Загруженность медицинских специализаций ({period_str})</div>
            </div>
        """, unsafe_allow_html=True)

        # --- БЛОК ТАБЛИЦЫ ---
        st.subheader("📋 Сводная таблица эффективности")
        # Вместо сырого HTML выводим через нативный интерактивный dataframe Streamlit
        st.dataframe(sp_report, use_container_width=True)

        # Палитра цветов для графиков
        PINK, TEAL = '#fce3ef', '#005F73'
        
        # --- ГРАФИКИ (Отрисовка в Streamlit) ---
        
        # Пример для p1 (Динамика по дням)
        st.subheader("1. Линейный график: Динамика использования времени")
        # Здесь должен быть твой код p1 = px.line(...) из скринов 6-7
        # Для демонстрации используем заглушку, замени на свой px.line, если настроен сбор по дням:
        if 'Дата' in df_clean.columns:
            df_daily = df_clean.groupby('Дата').sum().reset_index()
            p1 = px.line(df_daily, x='Дата', y='Занято записями', title="Динамика записей по дням")
            st.plotly_chart(p1, use_container_width=True)

        # График 4: ТОП по загрузке
        st.subheader("4. Горизонтальный Bar Chart (ТОП по Загрузке)")
        df_p4 = sp_report.sort_values('Загрузка %', ascending=True).copy()
        p4 = px.bar(
            df_p4, x='Загрузка %', y='Специализация', orientation='h',
            title="ТОП специализаций по Загрузке %", color='Загрузка %',
            color_continuous_scale=[[0.0, '#fce3ef'], [1.0, '#6A323A']],
            custom_data=['Табель', 'Занято записями']
        )
        p4.update_layout(xaxis_title="Загрузка расписания (%)", coloraxis_colorbar=dict(title="% загрузки"))
        p4.update_traces(hover_template="<b>%{y}</b><br>Загрузка: %{x:.1f}%<br>Выделено часов: %{customdata[0]:.1f} ч.<br>Занято записью: %{customdata[1]:.1f} ч.<extra></extra>")
        st.plotly_chart(p4, use_container_width=True)

        # График 5.1 & 5.2: Неявки
        st.subheader("5. Анализ неявок пациентов")
        col1, col2 = st.columns(2) # Делим экран на две колонки
        
        with col1:
            df_p51 = sp_report.sort_values('Потери', ascending=True).copy()
            p51 = px.bar(df_p51, x='Потери', y='Специализация', orientation='h', title="5.1. ТОП по Неявкам (часов)", color='Потери', color_continuous_scale=[[0.0, '#e6fcfb'], [1.0, '#005F73']])
            st.plotly_chart(p51, use_container_width=True)
            
        with col2:
            sp_report['Неявки %'] = (sp_report['Потери'] / sp_report['Занято записями'].clip(lower=1) * 100).round(1).fillna(0)
            df_p52 = sp_report.sort_values('Неявки %', ascending=True).copy()
            p52 = px.bar(df_p52, x='Неявки %', y='Специализация', orientation='h', title="5.2. ТОП по Неявкам (%)", color='Неявки %', color_continuous_scale=[[0.0, '#fce3ef'], [1.0, '#6A323A']])
            st.plotly_chart(p52, use_container_width=True)

        # График 6: Плиточная диаграмма (Treemap)
        st.subheader("6. Плиточная диаграмма: Объемы и явка")
        sp_report['Загрузка %'] = sp_report['Загрузка %'].fillna(0)
        sp_report['Явка %'] = sp_report['Явка %'].fillna(0)
        p6 = px.treemap(
            sp_report, path=['Специализация'], values='Табель', color='Явка %',
            color_continuous_scale=[[0.0, '#005F73'], [1.0, '#E0FFFF']],
            title="Объем выделенного времени и процент явки пациентов",
            custom_data=['Загрузка %', 'Явка %']
        )
        p6.update_traces(texttemplate="<b>%{label}</b><br>Выделено часов: %{value:.1f} ч.<br>Загрузка: %{customdata[0]:.1f}%<br>Явка: %{customdata[1]:.1f}%")
        st.plotly_chart(p6, use_container_width=True)

        # График 7: Матрица эффективности (Пузырьковая)
        st.subheader("7. Матрица эффективности: Загрузка и неявки")
        p7 = px.scatter(
            sp_report, x='Табель', y='Загрузка %', size='Табель', color='Неявки %',
            hover_name='Специализация', text='Специализация',
            title="Анализ загрузки и неявок по направлениям",
            color_continuous_scale=[[0.0, '#00A896'], [0.5, '#F4A261'], [1.0, '#D62828']]
        )
        p7.update_layout(template="plotly_white", xaxis_title="Выделено часов (ч.)", yaxis_title="Загрузка расписания (%)")
        st.plotly_chart(p7, use_container_width=True)

        # График 8: Каскадная диаграмма (Waterfall Balance)
        st.subheader("8. Каскадная диаграмма: Баланс рабочего времени")
        total_hours = sp_report['Табель'].sum()
        free_hours = sp_report['Свободно'].sum()
        lost_hours = sp_report['Потери'].sum()
        active_hours = sp_report['Дошло пациентов'].sum()
        
        p8 = go.Figure(go.Bar(
            x=["1. План по табелю", "Время без записи", "Неявки пациентов", "Фактически занято"],
            y=[total_hours, free_hours, lost_hours, active_hours],
            base=[0, total_hours - free_hours, total_hours - free_hours - lost_hours, 0],
            marker_color=['#005F73', '#B5838D', '#B5838D', '#6C9D9D']
        ))
        p8.update_layout(template="plotly_white", title="Баланс рабочего времени и структура потерь")
        st.plotly_chart(p8, use_container_width=True)

        # График 9: Кольцевая структура (Donut)
        st.subheader("9. Структура использования времени")
        p9 = go.Figure(data=[go.Pie(
            labels=['Фактически занято', 'Время без записи', 'Неявки пациентов'],
            values=[active_hours, free_hours, lost_hours],
            hole=.4, marker=dict(colors=['#6C9D9D', '#d1fff4', '#B5838D'])
        )])
        p9.update_layout(title="Структура использования рабочего времени докторов")
        st.plotly_chart(p9, use_container_width=True)

        # --- ТЕПЛОВЫЕ КАРТЫ за последние 14 дней (Графики 10 и 11) ---
        st.subheader("📅 Тепловые карты расписания (последние 14 дней)")
        
        # Повторяем логику фильтрации дат из скрина 14-16
        df_local = df_clean.copy()
        df_local["Parsed_Date"] = pd.to_datetime(df_local["Дата"], dayfirst=True, errors="coerce")
        df_local = df_local.dropna(subset=["Parsed_Date"])
        
        for col in ["Табель", "Занято записями", "Дошло пациентов"]:
            df_local[col] = pd.to_numeric(df_local[col].astype(str).str.replace(",", "."), errors="coerce")
            
        if not df_local.empty:
            last_date = df_local["Parsed_Date"].max()
            start_date = last_date - timedelta(days=13)
            df_local = df_local[(df_local["Parsed_Date"] >= start_date) & (df_local["Parsed_Date"] <= last_date)].copy()
            df_local["Дата"] = df_local["Parsed_Date"].dt.strftime("%d.%m.%Y")
            ordered_dates = [d.strftime("%d.%m.%Y") for d in pd.date_range(start=start_date, end=last_date, freq="D")]
            
            agg = df_local.groupby(["Дата", "Специализация"], as_index=False).agg({
                "Табель": "sum", "Занято записями": "sum", "Дошло пациентов": "sum"
            })
            
            agg["Загрузка %"] = np.where(agg["Табель"] > 0, (agg["Занято записями"] / agg["Табель"]) * 100, np.nan)
            h10 = agg.pivot(index="Дата", columns="Специализация", values="Загрузка %").reindex(ordered_dates)
            
            # Отрисовка Тепловой карты 10
            p10 = go.Figure(data=go.Heatmap(
