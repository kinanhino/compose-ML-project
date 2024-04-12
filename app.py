import os
from flask import Flask, request, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from joblib import load
import numpy as np
import pandas as pd

db_username = os.environ['DB_USERNAME']
db_password = os.environ['DB_PASSWORD']
db_name = os.environ['DB_NAME']
db_host = os.environ['DB_HOST']
db_port = os.environ['DB_PORT']
db_uri = f"postgresql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}"
print(f"Connecting db @{db_uri}")
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = db_uri
db = SQLAlchemy()
db.init_app(app)
class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(50), nullable=False)

    def __init__(self, name):
        self.name = name

# @app.route("/")
# def home():
#     return "Hello from my Containerized Server"

@app.route('/users', methods=['POST'])
def add_user():
    request_data = request.get_json()
    u_name = request_data['name']
    new_user = User(
        name=u_name)
    db.session.add(new_user)
    db.session.commit()
    return "User added successfully"

@app.route('/users')
def show_users():
    users = User.query.all()
    user_list = {}
    for user in users:
        user_list[user.id] = user.name
    return user_list

@app.route('/', methods=['GET', 'POST'])
def predict():
    lin_model = load('model/liner_pikel.pkl')
    if request.method == 'POST':
        # Extract the input values from the form
        year = int(request.form.get('Year'))
        obp = float(request.form.get('OBP'))
        ba = float(request.form.get('BA'))
        slg = float(request.form.get('SLG'))
        playoffs = int(request.form.get('Playoffs'))
        g = float(request.form.get('G'))
        oobp = float(request.form.get('OOBP'))
        oslg = float(request.form.get('OSLG'))
        league = int(request.form.get('League'))
        print(league)
        saved_values = {'league': f'{league}','oslg': f'{oslg}','oobp': f'{oobp}','g': f'{g}','playoffs': f'{playoffs}','slg': f'{slg}','year': f'{year}', 'obp': f'{obp}', 'slg': f'{slg}', 'ba':f'{ba}'}
        # Prepare the features
        #features = np.array([[year, obp, slg, ba,playoffs,g,oobp,oslg,league]])
        features = pd.DataFrame([[year, obp, slg, ba, playoffs, g, oobp, oslg, league]],
                           columns=['Year', 'OBP', 'SLG', 'BA', 'Playoffs', 'G', 'OOBP', 'OSLG', 'League_NL'])

        # Scale and transform the features


        # Predict the price
        rd_predicted = lin_model.predict(features)
        print(rd_predicted)
        print(type(rd_predicted))
        # Return the result to the same page
        return render_template('index.html',saved_values=saved_values ,prediction=round(rd_predicted[0], 2))
    saved_values = {'league': '', 'oslg': '', 'oobp': '', 'g': '','playoffs': '','slg': '','year': '', 'obp': '', 'slg': '', 'ba':''}

    return render_template('index.html', saved_values=saved_values, prediction=None)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5555)







