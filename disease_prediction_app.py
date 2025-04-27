import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
st.markdown("""
    <style>
        html, body, [class*="stApp"], .main-container {
            background: linear-gradient(to right, #ff9a9e, #fad0c4) !important;
            color: black !important;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown('<p class="big-title" style="font-size: 60px;">Disease Prediction App</p>', unsafe_allow_html=True) 
st.markdown('</div>', unsafe_allow_html=True)
# Load datasets
@st.cache_data
def load_data():
    liver_data = pd.read_csv("C:/Users/JANANI V/Desktop/MDP/Indian_Liver_patient.csv")
    kidney_data = pd.read_csv("C:/Users/JANANI V/Desktop/MDP/kidney_disease.csv")
    parkinsons_data = pd.read_csv("C:/Users/JANANI V/Desktop/MDP/parkinsons.csv")
    return liver_data, kidney_data, parkinsons_data

liver_data, kidney_data, parkinsons_data = load_data()

# Preprocess liver dataset
def preprocess_liver_data(data):
    data.columns = [col.lower().replace(" ", "_") for col in data.columns]
    data['dataset'] = data['dataset'].replace({1: 0, 2: 1})

    data = data.drop(columns=['gender'])
    num_cols = ['age', 'total_bilirubin', 'direct_bilirubin', 'alkaline_phosphotase',
                'alamine_aminotransferase', 'aspartate_aminotransferase', 'total_protiens', 'albumin']
    scaler = StandardScaler()
    data[num_cols] = scaler.fit_transform(data[num_cols])
    return data

# Preprocess kidney dataset
def preprocess_kidney_data(data):
    num_cols = ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
    cat_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    for col in num_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')
        data[col] = data[col].fillna(data[col].mean())
    for col in cat_cols:
        data[col] = data[col].fillna(data[col].mode()[0])
    label_encoder = LabelEncoder()
    for col in cat_cols:
        data[col] = label_encoder.fit_transform(data[col])
    data['classification'] = data['classification'].map({'ckd': 1, 'notckd': 0})

    return data

# Preprocess Parkinson's dataset
def preprocess_parkinsons_data(data):
    data.columns = [col.lower().replace(" ", "_") for col in data.columns]
    data = data.drop(columns=['name'])
    return data

liver_data = preprocess_liver_data(liver_data)
kidney_data = preprocess_kidney_data(kidney_data)
parkinsons_data = preprocess_parkinsons_data(parkinsons_data)

# Streamlit app

st.markdown("""
    <style>
        .stSidebar {
            background-color: #fff;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("Select Disease")

disease = st.sidebar.selectbox("Choose a disease to predict:", ["Liver Disease", "Kidney Disease", "Parkinson's Disease"])

if disease == "Liver Disease":
    st.header("Liver Disease Prediction")
    X = liver_data.drop(columns=['dataset'])
    y = liver_data['dataset']
    # Fill NaN values before SMOTE
    X = X.fillna(X.mean())  # Fill missing values with column mean
    smote = SMOTE(sampling_strategy=1, random_state=42)
    X_smote, y_smote = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)
    
    model_choice = st.sidebar.selectbox("Choose Model:", ["Random Forest", "XGBoost"])
    if model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=200, random_state=42)
    elif model_choice == "XGBoost":
        model = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42)
    

    model.fit(X_train, y_train)
    user_input = {col: st.number_input(f"Enter {col.replace('_', ' ').capitalize()}") for col in X.columns}
    
    user_df = pd.DataFrame([user_input])
    if user_df.sum().sum() == 0:
        st.info("🟢 No Risk Detected: All inputs are zero.")
    else:

        prediction = model.predict(user_df)
        user_probs = model.predict_proba(user_df)[:, 1]

        st.metric("Prediction Confidence", f"{user_probs[0]*100:.2f}%")
        if prediction[0] == 1:
            st.error("🔴 **High Risk: Liver Disease Detected**")
        else:
            st.success("🟢 **Low Risk: No Liver Disease**")

elif disease == "Kidney Disease":
    st.header("Kidney Disease Prediction")
    X = kidney_data.drop(columns=['classification'])
    y = kidney_data['classification']
    # Fill NaN values before SMOTE
    X = X.fillna(X.mean())  # Fill missing values with column mean
    smote = SMOTE(sampling_strategy=0.7, random_state=42)
    X_smote, y_smote = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)
    
    model_choice = st.sidebar.selectbox("Choose Model:", ["Random Forest", "XGBoost"])
    if model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=150,max_depth=5, class_weight="balanced", random_state=42)
    elif model_choice == "XGBoost":
        model = XGBClassifier(n_estimators=150, learning_rate=0.1, max_depth=3, scale_pos_weight=0.5,random_state=42)
    
    model.fit(X_train, y_train)
    user_input = {col: st.number_input(f"Enter {col.replace('_', ' ').capitalize()}") for col in X.columns}
    user_df = pd.DataFrame([user_input], columns=X.columns)
    if user_df.sum().sum() == 0:
        st.info("🟢 No Risk Detected: All inputs are zero.")
    else:
        scaler = StandardScaler().fit(X_train)  # Fit scaler on training data
        user_df_scaled = scaler.transform(user_df)  # Scale user input before prediction
        prediction = model.predict(user_df_scaled)
        user_probs = model.predict_proba(user_df_scaled)[:, 1]

        st.metric("Prediction Confidence", f"{user_probs[0]*100:.2f}%")
        if prediction[0] == 1:
            st.error("🔴 **High Risk: Kidney Disease Detected**")
        else:
            st.success("🟢 **Low Risk: No Kidney Disease**")

elif disease == "Parkinson's Disease":
    st.header("Parkinson's Disease Prediction")
    X = parkinsons_data.drop(columns=['status'])
    y = parkinsons_data['status']
    smote = SMOTE(sampling_strategy=0.7, random_state=42)
    X_smote, y_smote = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)
    
    model_choice = st.sidebar.selectbox("Choose Model:", ["Random Forest", "XGBoost"])
    if model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=200, random_state=42)
    elif model_choice == "XGBoost":
        model = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42)
    

    model.fit(X_train, y_train)
    user_input = {col: st.number_input(f"Enter {col.replace('_', ' ').capitalize()}") for col in X.columns}
    user_df = pd.DataFrame([user_input],columns= X.columns)
    if user_df.sum().sum() == 0:
        st.info("🟢 No Risk Detected: All inputs are zero.")
    else:
        scaler = StandardScaler().fit(X_train)  # Fit scaler on training data
        user_df_scaled = scaler.transform(user_df)  # Scale user input before prediction
        prediction = model.predict(user_df_scaled)
        user_probs = model.predict_proba(user_df_scaled)[:, 1]

        st.metric("Prediction Confidence", f"{user_probs[0]*100:.2f}%")
        if prediction[0] == 1:
            st.error("🔴 **High Risk: Parkinson’s Disease Detected**")
        else:
            st.success("🟢 **Low Risk: No Parkinson’s Disease**")