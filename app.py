import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)

model2 = pickle.load(open('claim_pred_model.pkl', 'rb'))

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
   if request.method == 'POST':
        features = [int(X) for X in request.form.values()]
        final_features = [np.array(features)]
        feature1 =[int(y) for y in request.form.values()]
        fin_feature1 = [np.array(feature1)]
        prediction = model2.predict(fin_feature1)
        if prediction==0:
         prediction ='No, person will not claim insurance'  
        elif prediction==1:
         prediction ='Yes, person can claim insurance'
        return render_template("result.html", prediction = prediction) 

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
    