import flask
import numpy as np
import pickle

model = pickle.load(open('model/regression_model.pkl', 'rb'))
app = flask.Flask(__name__, template_folder='templates')


@app.route('/')
def main():
    return(flask.render_template('home.html'))

@app.route('/predict', methods=['POST'])
def predict():
    # print('#', flask.request.form.values())
    int_features = [x for x in flask.request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    return(flask.render_template('home.html', prediction_value=prediction[0]))
    # return ('#', prediction)


if __name__ == '__main__':
    app.run()
