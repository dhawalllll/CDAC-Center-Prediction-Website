from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('indef.html')

@app.route('/predict', methods=['POST'])
def predict():
    rank = int(request.form['rank'])
    prediction = model.predict([[rank]])[0]
    return render_template('indef.html', rank=rank, result=prediction)

if __name__ == '__main__':
    app.run(debug=True)
