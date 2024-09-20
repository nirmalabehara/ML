import streamlit as st
import joblib
import pandas as pd

# Load your trained model
@st.cache_resource  # Caches the model to prevent reloading on every interaction
def load_model():
    return joblib.load('best_random_forest_model.pkl')

# Load model
model = load_model()

# Define input feature names (based on your model's training features)
feature_names = ['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 
                 'exng', 'oldpeak', 'slp', 'caa', 'thall']

# Define the Streamlit app layout
st.title('Heart Health Prediction App')

# Collect user input
def get_user_input():
    age = st.number_input('Age', min_value=1, max_value=120, value=50)
    sex = st.selectbox('Sex', [0, 1], help="0 = Female, 1 = Male")
    cp = st.number_input('Chest Pain Type (CP)', min_value=0, max_value=3, value=0)
    trtbps = st.number_input('Resting Blood Pressure (trtbps)', min_value=90, max_value=200, value=120)
    chol = st.number_input('Cholesterol (chol)', min_value=100, max_value=500, value=200)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', [0, 1])
    restecg = st.number_input('Resting Electrocardiographic Results (restecg)', min_value=0, max_value=2, value=0)
    thalachh = st.number_input('Maximum Heart Rate Achieved (thalachh)', min_value=70, max_value=220, value=150)
    exng = st.selectbox('Exercise Induced Angina (exng)', [0, 1])
    oldpeak = st.number_input('Oldpeak (ST depression induced by exercise)', min_value=0.0, max_value=10.0, value=1.0)
    slp = st.number_input('Slope of Peak Exercise ST Segment (slp)', min_value=0, max_value=2, value=1)
    caa = st.number_input('Number of Major Vessels (0-3) Colored by Flourosopy (caa)', min_value=0, max_value=3, value=0)
    thall = st.number_input('Thalassemia (thall)', min_value=0, max_value=3, value=1)

    # Create a dataframe for the inputs
    user_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        'trtbps': [trtbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalachh': [thalachh],
        'exng': [exng],
        'oldpeak': [oldpeak],
        'slp': [slp],
        'caa': [caa],
        'thall': [thall]
    })
    
    return user_data

# Collect input from the user
user_input = get_user_input()

# Display user input for verification
st.subheader('User Input:')
st.write(user_input)

# Make predictions
if st.button('Predict'):
    prediction = model.predict(user_input)
    st.subheader('Prediction Result:')
    
    if prediction[0] == 1:
        st.success("The model predicts that the patient is at risk of heart disease.")
    else:
        st.success("The model predicts that the patient is not at risk of heart disease.")
    
    st.write(f'Prediction: {prediction[0]}')

# To run Streamlit, use: streamlit run app.py
