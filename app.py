from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('template.html')


#@app.route('/predict',methods=['POST'])
#def predict():

#    int_features = [int(x) for x in request.form.values()]
 #   final_features = [np.array(int_features)]
   

  #  return render_template('template.html', prediction_text=final_features)

@app.route('/results',methods=['POST'])
def results():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)



if __name__ == "__main__":
    app.run(debug=True)