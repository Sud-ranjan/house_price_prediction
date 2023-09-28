import pickle
from flask import Flask,request,app,jsonify,url_for,render_template

import numpy as np
import pandas as pd

app = Flask(__name__)

# Loading the pickle model
model = pickle.load(open('housing_price_prediction_LR.pkl','rb'))
normalizer = pickle.load(open('normalizer.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods = ['POST'])
def predict_api():
    data = request.json['data']
    #  print(data)
    transformed_data = normalizer.transform(np.array(list(data.values())).reshape(1,-1))
    output = model.predict(transformed_data)
    print(output[0])

    return jsonify(output[0])

if __name__ == "__main__":
    app.run(debug=True)
