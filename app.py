# importing the necessary dependencies
# from sklearn.tree import DecisionTreeClassifier
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA

import pickle

import numpy as np
from flask import Flask, render_template, request
from flask_cors import cross_origin

app = Flask(__name__)  # initializing a flask app


@app.route('/', methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")


@app.route('/predict', methods=['POST', 'GET'])  # route to show the predictions in a web UI
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            steps = float(request.form['steps'])

            def parser(x):
                return datetime.strptime('19' + x, '%Y-%m')

            series = read_csv('https://raw.githubusercontent.com/blue-yonder/pydse/master/pydse/data/sales-of-shampoo-over-a-three-ye.csv', header=0, index_col=0, delimiter=';', parse_dates=True, squeeze=True, date_parser=parser)
            series.index = series.index.to_period('M')
            X = series.values
            size = int(len(X) * 0.66)
            train, test = X[0:size], X[size:len(X)]
            history = [x for x in train]
            predictions = list()
            for t in range(len(test)):
                model = ARIMA(history, order=(5, 1, 0))
                model_fit = model.fit()
                output = model_fit.forecast()
                yhat = output[0]
                predictions.append(yhat)
                obs = test[t]
                history.append(obs)
            mse = mean_squared_error(test, predictions)
            print("mean squared error is:", mse)

            # predictions using the loaded model file
            print('prediction is', yhat)
            # showing the prediction results in a UI

            render_template('results.html', result=yhat)
        except Exception as e:
            print('The Exception message is: ', e)
            return 'something is wrong'
    # return render_template('results.html')
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
# if __name__ == '__main__':
# app.run(host='0.0.0.0', port=8080)
# port = int(os.getenv("PORT"))
# host = '0.0.0.0'
# httpd = simple_server.make_server(host=host,port=5000, app=app)
# print("localhost:5000")
# httpd.serve_forever()
