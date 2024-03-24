import streamlit as st
import pandas as pd
import pickle
from custom_encoder import FrequencyEncoder

# Load preprocessor pipeline
with open("Preprocessor_pipeline.pkl", "rb") as file:
    preprocessor = pickle.load(file)

with open("cluster.pkl", "rb") as file:
    model = pickle.load(file)

def create_single_record():
    data = {
        'Number of Vehicles': st.number_input(f"Number of Vehicles involved in the accident:", min_value=0),
        'Age of Casualty': st.number_input(f"Age of person involved in the accident:", min_value=0),
        'Accident Counts': st.number_input(f"Count of accidents happened on that date:", min_value=0),
        'Road Surface': st.selectbox(f"Road Surface at the time of accident:", ['Dry', 'Wet / Damp', 'Frost / Ice', 'Snow', 'Flood']),
        'Lighting Conditions': st.selectbox(f"Lighting conditions at the time of accident:", ['Daylight', 'Darkness']),
        'Weather Conditions': st.selectbox(f"Weather conditions at the time of accident:", ['Normal', 'Raining', 'Windy', 'Snowing', 'Fog']),
        'Casualty Class': st.selectbox(f"Casualty class:", ['Driver', 'Passenger', 'Pedestrian', 'Vehicle or pillion passenger']),
        'Sex of Casualty': st.selectbox(f"Gender of person involved in the accident:", ['Male', 'Female']),
        'Type of Vehicle': st.selectbox(f"Type of vehicle involved in the accident:", ['Car', 'Truck', 'Bus', 'Pedal Cycle', 'Bike']),
        'Casualty Severity': st.selectbox(f"Casualty severity:", ['Fatal', 'Serious', 'Slight']),
        'Grid Ref: Easting': st.number_input(f"Grid Reference Easting:", min_value=0),
        'Grid Ref: Northing': st.number_input(f"Grid Reference Northing:", min_value=0)
    }
    return pd.DataFrame([data])

st.title("Single Record Input Form")

single_record_df = create_single_record()

st.write("\n## Generated Record:")
st.write(single_record_df)

# Preprocess the single record using preprocessor
preprocessed_data = preprocessor.transform(single_record_df)

st.write("\n## Preprocessed Data:")
st.write(preprocessed_data)

# model prediction
st.write("\n## Model Prediction:")
prediction = model.predict(preprocessed_data)

st.write("\n## Predicted Class:")
st.write(prediction)