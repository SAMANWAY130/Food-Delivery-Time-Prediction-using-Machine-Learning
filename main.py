import streamlit as st

import numpy as np
import pandas as pd

import os
import joblib


# 2. Re-run Haversine without rounding or int conversion
def calculate_haversine_distance(df):
    lat1, lon1 = np.radians(df['Restaurant_latitude']), np.radians(df['Restaurant_longitude'])
    lat2, lon2 = np.radians(df['Delivery_location_latitude']), np.radians(df['Delivery_location_longitude'])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * 6371.0





BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# NEW
model = joblib.load(os.path.join(BASE_DIR, "model(1) .pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler(1) .pkl"))
columns = joblib.load(os.path.join(BASE_DIR, "columns.pkl"))
Delivery_person_Age = st.number_input("Enter age")
Delivery_person_Ratings = st.number_input("Enter rating")

Restaurant_latitude = st.number_input("Enter restaurant latitude")
Restaurant_longitude = st.number_input("Enter restaurant longitude")

Delivery_location_latitude = st.number_input("Enter delivery latitude")
Delivery_location_longitude = st.number_input("Enter delivery longitude")

Weatherconditions = st.selectbox(
    "Choose weather",
    ["Sunny", "Stormy", "Sandstorms", "Cloudy", "Fog", "Windy"]
)

Road_traffic_density = st.selectbox(
    "Choose road traffic density",
    ["High", "Jam", "Low", "Medium"]
)

Vehicle_condition = st.selectbox(
    "Choose vehicle condition",
    [2, 0, 1, 3]
)

Type_of_order = st.selectbox(
    "Choose type of order",
    ["Snack", "Drinks", "Buffet", "Meal"]
)

Type_of_vehicle = st.selectbox("coose vehcle " ,['motorcycle', 'scooter', 'electric_scooter', 'bicycle'])
multiple_deliveries = st.selectbox("choose number of delivers " ,[0., 1., 3., 2.])
Festival = st.selectbox("select festival probab " ,["Yes","No"])
City = st.selectbox("select city " , ['Urban', 'Metropolitian', 'Semi-Urban'])
hour = st.number_input("1 - 24 hours")
    

st.title("CALCULATE ESTIMTATED TIME TO DELIVER")
predict = st.button("PREDICT")
if predict :
        df = pd.DataFrame({
        "Delivery_person_Age": [Delivery_person_Age],
        "Delivery_person_Ratings": [Delivery_person_Ratings],
        "Restaurant_latitude": [Restaurant_latitude],
        "Restaurant_longitude": [Restaurant_longitude],
        "Delivery_location_latitude": [Delivery_location_latitude],
        "Delivery_location_longitude": [Delivery_location_longitude],
        "Weatherconditions": [Weatherconditions],
        "Road_traffic_density": [Road_traffic_density],
        "Vehicle_condition": [Vehicle_condition],
        "Type_of_order": [Type_of_order],
        "Type_of_vehicle": [Type_of_vehicle],
        "multiple_deliveries": [multiple_deliveries],
        "Festival": [Festival],
        "City": [City],
        "hour" : [hour]

    })
        
        # 1. Make sure coordinate columns are floats
        df['Restaurant_latitude'] = df['Restaurant_latitude'].astype(float)
        df['Restaurant_longitude'] = df['Restaurant_longitude'].astype(float)
        df['Delivery_location_latitude'] = df['Delivery_location_latitude'].astype(float)
        df['Delivery_location_longitude'] = df['Delivery_location_longitude'].astype(float)
        
        df['distance_km'] = calculate_haversine_distance(df)
        df["Road_traffic_density"] = df["Road_traffic_density"].map({"Low":1,"Medium":2,"High":3,"Jam":4})
        df = pd.get_dummies(df,drop_first=True)
        df = df.reindex(columns=columns , fill_value=0)
        inpot_scae = scaler.transform(df)
        prediction = model.predict(inpot_scae)
        st.success(f"estimated time is {prediction} min")

        




