import pandas as pd
import pickle
import joblib
from preprosessing_file import preprossesing  # модуль, который присылал Данила

# пример данных (как у Данила)
raw_data = ['2021-11-24', '14:36:32', 1, 'ZpYIoDJMcFzVoPFsHGJL', 'banner', 'LEoPHuyFvzoNfnzGgfcd'
            , 'vCIpmpaGBnIQhyYNkXqp', 'mobile', 'other', '360x720', 'Chrome', 'Russia', 'other']
dct_path = 'dct.pkl' # словарь, который присылал Данила
with open(dct_path, 'rb') as f:
    loaded_dct = pickle.load(f)
processed = preprossesing(raw_data, loaded_dct) # препроцессинг, который присылал Данила (вылезает много warning, но работает)

rf2 = r_forest.RandomForestForHackathon() # модель Random Forest
y_pred5 = rf2.predict(processed)
y_pred_proba5 = rf2.predict_proba(processed)[:, 1]
print(y_pred5)
print(y_pred_proba5)