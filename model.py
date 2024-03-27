import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("/home/kareem/alex/model/alex-91eda-firebase-adminsdk-xv5r9-cc8546f491.json")
firebase_admin.initialize_app(cred, {
  "apiKey": "AIzaSyCy-2oljvofzDvmkE-9KMHCmC9MER7-khY",
  "authDomain": "alex-91eda.firebaseapp.com",
  "databaseURL": "https://alex-91eda-default-rtdb.firebaseio.com",
  "projectId": "alex-91eda",
  "storageBucket": "alex-91eda.appspot.com",
  "messagingSenderId": "642791113215",
  "appId": "1:642791113215:web:1b8b52e959b25146a6e7ca",
  "measurementId": "G-2RXMC54PXS"
})

ref = db.reference()

data = pd.read_csv('output_data.csv')

conditions = {
    0: "Normal",
    1: "Depression",
    2: "High Depression",
    3: "Mania",
    4: "High Mania"
}

def predict (_sleep_rate, _depression_score, _mania_score, _activity):
    Y = data["output"]
    X = data[['sleep_rate', 'depression_score', 'mania_score', 'activity']]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = tree.DecisionTreeClassifier()
    model.fit(X_train, Y_train)
    # predictions = model.predict(X_test)
    score = model.score(X_test, Y_test)
    prediction = model.predict([
        [_sleep_rate, _depression_score, _mania_score, _activity]
    ])[0]
    print ("the prediction is ", prediction)
    print ("the score is ", score)
    ref.child("model").update({
      "output": conditions[int(prediction)],
    })
    return (prediction)


while True:
    sleep_rate = ref.child("model").child("sleep_rate").get()
    activity = ref.child("model").child("activity").get()
    
    depression_score_parent = ref.child("scores").child("parent").child("depression").get()
    mania_score_parent = ref.child("scores").child("parent").child("mania").get()

    depression_score_patient = ref.child("scores").child("patient").child("depression").get()
    mania_score_patient = ref.child("scores").child("patient").child("mania").get()

    predict(sleep_rate, int((depression_score_parent + depression_score_patient)/2), int((mania_score_parent + mania_score_patient)/2), activity)