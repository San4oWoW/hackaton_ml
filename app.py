from preprosessing_file import preprossesing
from flask import Flask, request, jsonify
from catboost import CatBoostClassifier
import pandas as pd
import pickle



model_path = 'cat_boost_balanced.cbm'
catboost_model = CatBoostClassifier()
catboost_model.load_model(model_path)


dct_path = 'dct.pkl'
with open(dct_path, 'rb') as f:
    loaded_dct = pickle.load(f)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    input_json = request.get_json()
    raw_str = input_json['data']
    raw_data = raw_str.split()  # теперь это ["10", "2", "15"]
    processed = preprossesing(raw_data, loaded_dct)
    prediction = catboost_model.predict(processed)
    result = int(prediction[0])

    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)




