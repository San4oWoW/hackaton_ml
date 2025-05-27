import pandas as pd
from catboost import CatBoostClassifier

# Настройка модели
model = CatBoostClassifier()
model.load_model("cat_boost.cbm")
model.eval = lambda: None  # для единообразия интерфейса с PyTorch-моделью (если потребуется)


# Названия признаков — порядок должен соответствовать обучению
FEATURE_NAMES = [
    'visit_date', 'visit_time', 'visit_number', 'utm_source', 'utm_medium',
    'utm_campaign', 'utm_adcontent', 'device_category', 'device_brand',
    'device_screen_resolution', 'device_browser', 'geo_country', 'geo_city'
]


# Если подается только строка
def predict_from_dataframe_row(row):
    data_row = pd.DataFrame([row], columns=FEATURE_NAMES)
    return int(model.predict(data_row)[0])


# Если из csv файла
def predict_from_csv(path):
    data = pd.read_csv(path)
    return [int(i) for i in model.predict(data[FEATURE_NAMES])]


