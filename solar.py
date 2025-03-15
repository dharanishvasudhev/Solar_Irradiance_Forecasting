import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import time
import datetime
import MySQLdb

# Database connection
db = MySQLdb.connect(host="localhost", user="root", passwd="WillofD@30", db="prediction")
cur = db.cursor()

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

# Real-time prediction loop
while True:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fxn()

    # Load dataset
    df = pd.read_csv("C:/Users/Lenovo/Desktop/AIML/Solar Prediction/mdu_data.csv")
    df = df.fillna(0)
    nonzero_mean = df[df != 0].mean()

    cols = [0, 1, 2, 3, 4]a
    X = df[df.columns[cols]].values
    
    Y_temp = df[df.columns[5]].values.ravel()
    Y_ghi = df[df.columns[6]].values.ravel()

    # Split dataset
    x_train, x_test, y_temp_train, y_temp_test = train_test_split(X, Y_temp, random_state=42)
    x_train, x_test, y_ghi_train, y_ghi_test = train_test_split(X, Y_ghi, random_state=42)

    # Train models
    temp_model = RandomForestRegressor()
    ghi_model = RandomForestRegressor()

    temp_model.fit(x_train, y_temp_train)
    ghi_model.fit(x_train, y_ghi_train)

    # Real-time prediction
    current_time = datetime.datetime.now() + datetime.timedelta(minutes=15)
    time_updated = current_time.strftime("%Y-%m-%d %H:%M")

    now = current_time.strftime("%Y,%m,%d,%H,%M")
    now = [int(i) for i in now.split(",")]

    temp = temp_model.predict([now])
    ghi = ghi_model.predict([now])

    # Power calculation
    f = 0.18 * 7.4322 * ghi
    insi = 0.05 * (temp - 25)
    midd = 1 - insi
    power = f * midd

    # Accuracy calculation
    temp_predictions = temp_model.predict(x_test)
    ghi_predictions = ghi_model.predict(x_test)

    temp_accuracy = max(0, min(100, r2_score(y_temp_test, temp_predictions) * 100))
    ghi_accuracy = max(0, min(100, r2_score(y_ghi_test, ghi_predictions) * 100))

    # Print predictions and accuracy
    print(f"Prediction at {time_updated}:")
    print(f"  Temperature: {temp[0]:.2f}°C (Accuracy: {temp_accuracy:.2f}%)")
    print(f"  GHI: {ghi[0]:.2f} W/m² (Accuracy: {ghi_accuracy:.2f}%)")
    print(f"  Power: {power[0]:.2f} W")

    # Insert into database
    sql = """INSERT INTO power_prediction (time_updated, Temperature, GHI, power) VALUES (%s, %s, %s, %s)"""
    try:  
        print("Writing to the database...")  
        cur.execute(sql, (time_updated, temp[0], ghi[0], power[0]))  
        db.commit()  
        print("Write complete")  
    except Exception as e:  
        db.rollback()  
        print("We have a problem:", e)

    time.sleep(1)

cur.close()
db.close()
