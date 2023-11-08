from flask import Flask, render_template
import numpy as np


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('template.html')


@app.route('/predict',methods=['POST'])
def predict():

    return render_template('template.html', prediction_text='YES')


if __name__ == "__main__":
    app.run(debug=True)