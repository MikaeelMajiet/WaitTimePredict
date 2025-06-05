import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

st.set_page_config(layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv(r"C:\Users\mikae\Downloads\SAND PROJECT\clinic_data.csv")

@st.cache_resource
def load_models():
    lr = joblib.load(r"C:\Users\mikae\Downloads\SAND PROJECT\lr_model.joblib")
    rf = joblib.load(r"C:\Users\mikae\Downloads\SAND PROJECT\rf_model.joblib")
    return lr, rf

df = load_data()
lr_model, rf_model = load_models()

# Categorical options from your dataset - adjust if needed
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
time_of_day_options = sorted(df['time_of_day'].unique())
urban_rural_options = sorted(df['urban_rural'].unique())
weather_options = sorted(df['weather'].unique())
clinic_type_options = sorted(df['clinic_type'].unique())
appointment_type_options = sorted(df['appointment_type'].unique())
walk_in_policy_options = sorted(df['walk_in_policy'].unique())

st.title("Clinic Appointment Scheduling - Predict Wait Times")

st.sidebar.header("Select Model and Input Features")

model_choice = st.sidebar.selectbox("Choose Model", options=['Random Forest', 'Linear Regression'])

def user_inputs():
    inputs = dict()
    inputs['day_of_week'] = st.sidebar.selectbox("Day of Week", options=days_order, index=0)
    inputs['time_of_day'] = st.sidebar.selectbox("Time of Day", options=time_of_day_options, index=0)
    inputs['urban_rural'] = st.sidebar.selectbox("Urban or Rural Clinic", options=urban_rural_options, index=0)
    inputs['weather'] = st.sidebar.selectbox("Weather", options=weather_options, index=0)
    inputs['clinic_type'] = st.sidebar.selectbox("Clinic Type", options=clinic_type_options, index=0)
    inputs['appointment_type'] = st.sidebar.selectbox("Appointment Type", options=appointment_type_options, index=0)
    inputs['walk_in_policy'] = st.sidebar.selectbox("Walk-in Policy", options=walk_in_policy_options, index=0)
    
    inputs['cancellations'] = st.sidebar.slider("Number of Cancellations", 0, 20, 2)
    inputs['cancellation_rate'] = st.sidebar.slider("Cancellation Rate", 0.0, 1.0, 0.1, step=0.01)
    inputs['no_shows'] = st.sidebar.slider("Number of No-Shows", 0, 20, 3)
    inputs['no_show_rate'] = st.sidebar.slider("No-Show Rate", 0.0, 1.0, 0.1, step=0.01)
    
    inputs['staff_count'] = st.sidebar.slider("Total Staff Count", 1, 50, 10)
    inputs['staff_doctors'] = st.sidebar.slider("Number of Doctors", 0, inputs['staff_count'], 3)
    inputs['staff_nurses'] = st.sidebar.slider("Number of Nurses", 0, inputs['staff_count'], 5)
    
    inputs['clinic_open_hours'] = st.sidebar.slider("Clinic Open Hours", 1, 24, 8)
    inputs['appointment_lead_time'] = st.sidebar.slider("Appointment Lead Time (days)", 0, 30, 5)
    
    inputs['power_outage'] = st.sidebar.selectbox("Power Outage?", options=[0,1], format_func=lambda x: "No" if x==0 else "Yes")
    inputs['equipment_downtime'] = st.sidebar.selectbox("Equipment Downtime?", options=[0,1], format_func=lambda x: "No" if x==0 else "Yes")
    
    inputs['appointments'] = st.sidebar.slider("Number of Appointments", 0, 200, 60)
    inputs['walk_ins'] = st.sidebar.slider("Expected Walk-ins", 0, 50, 10)
    inputs['emergencies'] = st.sidebar.slider("Number of Emergencies", 0, 10, 1)
    
    inputs['avg_appointment_duration'] = st.sidebar.slider("Avg Appointment Duration (mins)", 5, 60, 20)
    inputs['triage_score_avg'] = st.sidebar.slider("Avg Triage Score", 1.0, 5.0, 2.5, step=0.1)
    inputs['transport_access_score'] = st.sidebar.slider("Transport Access Score", 0.0, 1.0, 0.8, step=0.01)
    
    return inputs

inputs = user_inputs()

# Calculate derived features
patients_served_est = max(inputs['appointments'] - int(inputs['appointments'] * inputs['cancellation_rate']) - int(inputs['appointments'] * inputs['no_show_rate']) + inputs['walk_ins'], 1)
staff_per_patient = inputs['staff_count'] / patients_served_est if patients_served_est > 0 else 0
efficiency_score = (patients_served_est / inputs['staff_count']) - 10 if inputs['staff_count'] > 0 else -10

feature_dict = {
    'day_of_week': inputs['day_of_week'],
    'time_of_day': inputs['time_of_day'],
    'urban_rural': inputs['urban_rural'],
    'weather': inputs['weather'],
    'clinic_type': inputs['clinic_type'],
    'appointment_type': inputs['appointment_type'],
    'walk_in_policy': inputs['walk_in_policy'],
    'cancellations': inputs['cancellations'],
    'cancellation_rate': inputs['cancellation_rate'],
    'no_shows': inputs['no_shows'],
    'no_show_rate': inputs['no_show_rate'],
    'staff_count': inputs['staff_count'],
    'staff_doctors': inputs['staff_doctors'],
    'staff_nurses': inputs['staff_nurses'],
    'clinic_open_hours': inputs['clinic_open_hours'],
    'appointment_lead_time': inputs['appointment_lead_time'],
    'power_outage': inputs['power_outage'],
    'equipment_downtime': inputs['equipment_downtime'],
    'appointments': inputs['appointments'],
    'walk_ins': inputs['walk_ins'],
    'emergencies': inputs['emergencies'],
    'avg_appointment_duration': inputs['avg_appointment_duration'],
    'triage_score_avg': inputs['triage_score_avg'],
    'transport_access_score': inputs['transport_access_score'],
    'staff_per_patient': staff_per_patient,
    'efficiency_score': efficiency_score
}

# Convert to DataFrame
input_df = pd.DataFrame([feature_dict])

# Predict function helper (handling categorical encoding by model pipeline)
def predict_wait_time(df, model):
    return model.predict(df)[0]

wait_time_pred = None
if model_choice == 'Random Forest':
    wait_time_pred = predict_wait_time(input_df, rf_model)
else:
    wait_time_pred = predict_wait_time(input_df, lr_model)

st.header(f"Predicted Average Wait Time: {wait_time_pred:.2f} minutes using {model_choice}")

# Generate dynamic heatmap data by varying day_of_week and time_of_day, keeping other inputs fixed
heatmap_rows = []
for day in days_order:
    for time in time_of_day_options:
        row = feature_dict.copy()
        row['day_of_week'] = day
        row['time_of_day'] = time
        heatmap_rows.append(row)

heatmap_df = pd.DataFrame(heatmap_rows)

def batch_predict(df, model):
    preds = model.predict(df)
    return preds

heatmap_df['predicted_wait'] = batch_predict(heatmap_df, rf_model if model_choice == 'Random Forest' else lr_model)

heatmap_pivot = heatmap_df.pivot(index='day_of_week', columns='time_of_day', values='predicted_wait')
heatmap_pivot = heatmap_pivot.reindex(index=days_order)  # Ensure day order

colorscale = [
    [0.0, "green"],
    [0.5, "yellow"],
    [1.0, "red"]
]

fig = px.imshow(
    heatmap_pivot,
    color_continuous_scale=colorscale,
    labels=dict(x="Time of Day", y="Day of Week", color="Predicted Avg Wait Time (mins)"),
    aspect="auto"
)

st.subheader("Predicted Wait Time Heatmap (Green = Low Wait, Red = High Wait)")
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("### How to run this app:")
st.code("streamlit run app.py")
