from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import pickle
app=Flask(__name__,template_folder="templates")
model=pickle.load(open('PAE_model.pkl','rb'))
@app.route('/')
def form():
    return render_template("form.html")
@app.route('/predict',methods=["POST","GET"])
def predict():
    input_features=[float(x) for x in request.form.values()]
    features_value=[np.array(input_features)]
    print(features_value)
    features_name=['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','Department','salary']
    scaler=pickle.load(open("scaler.pkl","rb"))
    x_test_scaled=scaler.transform(features_value)
    prediction=model.predict(x_test_scaled)
    output=prediction[0]
    return render_template('resultEA.html',prediction_output=output)
if __name__=="__main__":
    app.run(port=5000,debug=False)