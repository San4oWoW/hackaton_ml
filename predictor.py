import pickle
import pandas as pd


def load_model_and_encoders(model_path="xgb_model.pkl", encoders_path="label_encoders.pkl"):
    """
    Загружает модель XGBoost и словарь LabelEncoder'ов
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(encoders_path, "rb") as f:
        encoders = pickle.load(f)
    return model, encoders


def predict_from_input(input_dict, model, encoders):
    """
    Делает предсказание по одному словарю с признаками
    """
    df_input = pd.DataFrame([input_dict])

    # Кодируем категориальные переменные
    for col, le in encoders.items():
        if col in df_input.columns:
            df_input[col] = le.transform(df_input[col].astype(str))
        else:
            df_input[col] = 0  # Если признак отсутствует — заполняем 0

    # Убедимся, что все нужные признаки есть
    for col in model.get_booster().feature_names:
        if col not in df_input.columns:
            df_input[col] = 0

    df_input = df_input[model.get_booster().feature_names]

    proba = model.predict_proba(df_input)[:, 1]
    return proba[0]

# - `session_id` — ID визита
# - `client_id` — ID посетителя
# - `visit_date` — дата визита
# - `visit_time` — время визита
# - `visit_number` — порядковый номер визита клиента
# - `utm_source` — канал привлечения
# - `utm_medium` — тип привлечения
# - `utm_campaign` — рекламная кампания
# - `utm_keyword` — ключевое слово
# - `device_category` — тип устройства

# - `session_id` — ID визита
# - `hit_date` — дата события
# - `hit_time` — время события
# - `hit_number` — порядковый номер события в рамках сессии
# - `hit_type` — тип события
# - `hit_referer` — источник события
# - `hit_page_path` — страница события
# - `event_category` — тип действия
# - `event_action` — действие
# - `event_label` — тег действия
# - `event_value` — значение результата действия
