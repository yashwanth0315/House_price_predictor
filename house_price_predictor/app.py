import pandas as pd
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load data and model
data = pd.read_csv('cleaned_data.csv')
pipe = pickle.load(open('ridgemodel.pkl', 'rb'))

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = int(request.form.get('bhk'))
    bath = int(request.form.get('bath'))
    sqft = float(request.form.get('sqft'))

    print(location, bhk, bath, sqft)

    input_df = pd.DataFrame([[location, sqft, bath, bhk]],
                            columns=['location', 'total_sqft', 'bath', 'bhk'])

    prediction = pipe.predict(input_df)[0] * 100000
    print(input_df)
    print(prediction)

    return str(round(prediction, 2))

if __name__ == "__main__":
    app.run(debug=True, port=5001)
