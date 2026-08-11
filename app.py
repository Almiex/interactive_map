# Сохраните как patch_app.py и запустите: python patch_app.py

with open('app_v8.py', 'r', encoding='utf-8') as f:
    code = f.read()

# 1. BLOCK_WEIGHTS
code = code.replace(
    '''BLOCK_WEIGHTS = {
    "demand": 0.27,
    "accessibility": 0.22,
    "traffic": 0.15,
    "parking": 0.07,
    "competition": 0.11,
    "medical_ecosystem": 0.10,
    "environment": 0.08,
}''',
    '''BLOCK_WEIGHTS = {
    "demand": 0.22,
    "accessibility": 0.15,
    "traffic": 0.18,
    "parking": 0.20,
    "competition": 0.10,
    "medical_ecosystem": 0.08,
    "environment": 0.07,
}'''
)

# 2. FACTOR_WEIGHTS — полная замена
old_fw = code[code.find('FACTOR_WEIGHTS = {'):code.find('FACTOR_BLOCKS = {}')]
new_fw = '''FACTOR_WEIGHTS = {
    "population_500m": 0.12,
    "population_1km": 0.15,
    "target_population_share": 0.18,
    "target_population_count_1km": 0.15,
    "income_fit": 0.15,
    "age_fit": 0.12,
    "gender_fit": 0.10,
    "residential_density": 0.03,

    "walk_5min": 0.25,
    "car_10min": 0.30,
    "public_transport_access": 0.20,
    "physical_barriers": 0.25,

    "pedestrian_traffic_quality": 0.20,
    "car_traffic_quality": 0.15,
    "traffic_target_share": 0.20,
    "traffic_time_fit": 0.15,
    "residential_traffic_share": 0.15,
    "commercial_traffic_share": 0.05,
    "medical_traffic_share": 0.05,
    "visibility": 0.03,
    "wayfinding": 0.02,

    "parking_supply": 0.22,
    "parking_distance": 0.18,
    "free_parking": 0.12,
    "parking_competition": 0.18,
    "parking_time_fit": 0.15,
    "dropoff_access": 0.10,
    "parking_reliability": 0.05,

    "competitor_density": 0.20,
    "competitor_strength": 0.20,
    "competitor_distance": 0.10,
    "competitive_capacity": 0.20,
    "price_level_fit": 0.10,
    "market_saturation": 0.15,
    "market_gap": 0.05,

    "hospital_synergy": 0.25,
    "specialist_synergy": 0.35,
    "medical_cluster": 0.25,
    "healthcare_traffic": 0.15,

    "residential_commercial_balance": 0.15,
    "home_clinic_environment": 0.30,
    "information_noise": 0.10,
    "noise_environment": 0.10,
    "safety_environment": 0.10,
    "pedestrian_comfort": 0.05,
    "daily_services": 0.10,
    "office_dependence_risk": 0.10,
}

FACTOR_BLOCKS = {}
'''

code = code.replace(old_fw, new_fw)

# 3. FACTOR_BLOCKS
old_fb = code[code.find('for f in [\n    "population_500m"'):code.find('FACTOR_NAMES = {')]
new_fb = '''for f in [
    "population_500m", "population_1km",
    "target_population_share", "target_population_count_1km",
    "income_fit", "age_fit", "gender_fit", "residential_density"
]:
    FACTOR_BLOCKS[f] = "demand"

for f in [
    "walk_5min", "car_10min", "public_transport_access", "physical_barriers"
]:
    FACTOR_BLOCKS[f] = "accessibility"

for f in [
    "pedestrian_traffic_quality", "car_traffic_quality",
    "traffic_target_share", "traffic_time_fit", "residential_traffic_share",
    "commercial_traffic_share", "medical_traffic_share",
    "visibility", "wayfinding"
]:
    FACTOR_BLOCKS[f] = "traffic"

for f in [
    "parking_supply", "parking_distance", "free_parking",
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
    "hospital_synergy", "specialist_synergy", "medical_cluster",
    "healthcare_traffic"
]:
    FACTOR_BLOCKS[f] = "medical_ecosystem"

for f in [
    "residential_commercial_balance", "home_clinic_environment",
    "information_noise", "noise_environment", "safety_environment",
    "pedestrian_comfort", "daily_services", "office_dependence_risk"
]:
    FACTOR_BLOCKS[f] = "environment"

'''

code = code.replace(old_fb, new_fb)

# 4. Удаляем поля из GeoAIProfile
for field in [
    '    population_3km: int = Field(ge=0, le=3000000)\n',
    '    population_growth: int = Field(ge=0, le=100)\n',
    '    family_profile: int = Field(ge=0, le=100)\n',
    '    daytime_population_balance: int = Field(ge=0, le=100)\n',
    '    transit_connectivity: int = Field(ge=0, le=100)\n',
    '    road_connectivity: int = Field(ge=0, le=100)\n',
    '    pedestrian_connectivity: int = Field(ge=0, le=100)\n',
    '    office_traffic_share: int = Field(ge=0, le=100)\n',
    '    family_services: int = Field(ge=0, le=100)\n',
    '    fitness_services: int = Field(ge=0, le=100)\n',
]:
    code = code.replace(field, '')

# 5. Убираем population_3km из нормализации
code = code.replace(
    '    elif factor == "population_3km":\n        value = normalize_population(value, 50000, 400000)\n',
    ''
)

# 6. Усиливаем hard penalties
code = code.replace(
    '''    penalty = min(penalty, 35.0)
    final = round(clamp(absolute_score - penalty), 1)

    return final, penalty''',
    '''    # Комбинированный штраф за "убийственную" комбинацию
    if (profile.get("parking_supply", 0) <= 15 and 
        profile.get("home_clinic_environment", 0) <= 25):
        penalty += 10  # БЦ без парковки = катастрофа

    penalty = min(penalty, 50.0)
    final = round(clamp(absolute_score - penalty), 1)

    return final, penalty'''
)

# 7. Усиливаем hard barriers
old_hard = 'def calculate_hard_barriers(profile: dict) -> List[str]:'
idx_hard = code.find(old_hard)
idx_hard_end = code.find('def apply_hard_penalties', idx_hard)
old_hard_body = code[idx_hard:idx_hard_end]

new_hard_body = '''def calculate_hard_barriers(profile: dict) -> List[str]:
    barriers = []

    if profile.get("physical_barriers", 0) >= 85:
        barriers.append("Высокий уровень физических барьеров.")

    if profile.get("parking_supply", 0) <= 10:
        barriers.append("КАТАСТРОФА: отсутствие парковки. Пациенты не смогут приехать на авто.")
    elif profile.get("parking_supply", 0) <= 20:
        barriers.append("КРИТИЧЕСКИ НИЗКАЯ парковочная ёмкость.")

    if profile.get("parking_reliability", 0) <= 10:
        barriers.append("Парковка отсутствует или занята постоянно.")

    if profile.get("home_clinic_environment", 0) <= 20:
        barriers.append("Среда НЕ соответствует формату «клиника у дома». Вероятно, БЦ или промзона.")

    if profile.get("gender_fit", 0) <= 30:
        barriers.append("Половой состав резко не соответствует ЦА. Вероятно, БЦ/промзона с мужским преобладанием.")

    if profile.get("traffic_target_share", 0) <= 30:
        barriers.append("Трафик практически не соответствует ЦА.")

    if profile.get("office_dependence_risk", 0) >= 80:
        barriers.append("ВЫСОКИЙ РИСК зависимости от офисного трафика. Вечером и выходные — пусто.")

    if profile.get("wayfinding", 0) <= 20:
        barriers.append("Слабая навигационная понятность объекта.")

    return barriers

'''

code = code[:idx_hard] + new_hard_body + code[idx_hard_end:]

with open('app_final.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("✅ Готово: app_final.py")
