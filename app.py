from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('template.html')


@app.route('/predict',methods=['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    print("FEATURES",request.form.values())
    final_features = [np.array(int_features)]
    print("FINAL",final_features)
    prediction = model.predict(final_features)
    estado=""
    if(prediction==0):
        estado="Muerto"
    else:
        estado="Vivito y coleando"

    return render_template('template.html', prediction_text='El pasajero está: {}'.format(estado))

@app.route('/results',methods=['POST'])
def results():
    data = request.get_json(force=True)
    print("DATA",data)
    prediction = model.predict([np.array(list(data.values()))])
    print("PREDICTION",prediction)
    output = prediction[0]
    print("OUTPUT",output)
    if(output==0):
        output="HA MUERTO"
    else:
        output="NO HA MUERTO"
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)