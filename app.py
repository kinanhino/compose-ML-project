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

class ModelPrediction(db.Model):
    __tablename__ = 'predictiontbl'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    oslg = db.Column(db.Float)
    oobp = db.Column(db.Float)
    playoffs = db.Column(db.Float)
    obp = db.Column(db.Float)
    slg = db.Column(db.Float)
    year = db.Column(db.Float)
    g = db.Column(db.Float)
    league_nl = db.Column(db.Float)
    ba = db.Column(db.Float)
    predict = db.Column(db.Float)

    def __init__(self, oobp, playoffs, obp, slg, year, g, league_nl, oslg, ba, predict):
        self.oobp = oobp
        self.oslg = oslg
        self.playoffs = playoffs
        self.obp = obp
        self.slg = slg
        self.year = year
        self.g = g
        self.league_nl = league_nl
        self.ba = ba
        self.predict = predict

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
        # extract the input values from the form
        year = int(request.form.get('Year'))
        obp = float(request.form.get('OBP'))
        ba = float(request.form.get('BA'))
        slg = float(request.form.get('SLG'))
        playoffs = int(request.form.get('Playoffs'))
        g = float(request.form.get('G'))
        oobp = float(request.form.get('OOBP'))
        oslg = float(request.form.get('OSLG'))
        league = int(request.form.get('League'))
        saved_values = {'league': f'{league}','oslg': f'{oslg}','oobp': f'{oobp}','g': f'{g}','playoffs': f'{playoffs}','slg': f'{slg}','year': f'{year}', 'obp': f'{obp}', 'slg': f'{slg}', 'ba':f'{ba}'}
        
        # prepare the features
        features = pd.DataFrame([[year, obp, slg, ba, playoffs, g, oobp, oslg, league]],
                           columns=['Year', 'OBP', 'SLG', 'BA', 'Playoffs', 'G', 'OOBP', 'OSLG', 'League_NL'])

        # predict the price
        rd_predicted = lin_model.predict(features)
        rd_predicted = round(rd_predicted[0], 2)
        saved_values['rd'] = rd_predicted
        # save all of the parameters to the database
        save_prediction_to_db(saved_values)

        # return the result to the same page
        return render_template('index.html',saved_values=saved_values ,prediction=rd_predicted)
    
    saved_values = {'league': '', 'oslg': '', 'oobp': '', 'g': '','playoffs': '','slg': '','year': '', 'obp': '', 'slg': '', 'ba':''}
    return render_template('index.html', saved_values=saved_values, prediction=None)

def save_prediction_to_db(db_dict):
    new_prediction = ModelPrediction(
        year=db_dict['year'],
        obp=db_dict['obp'],
        slg=db_dict['slg'],
        ba=db_dict['ba'],
        playoffs=db_dict['playoffs'],
        g=db_dict['g'],
        oobp=db_dict['oobp'],
        oslg=db_dict['oslg'],
        league_nl=db_dict['league'],
        predict=db_dict['rd'])

    db.session.add(new_prediction)
    db.session.commit()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5555)







