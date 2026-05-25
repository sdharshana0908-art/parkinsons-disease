
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Parkinson Disease Prediction",
    page_icon="🩺",
    layout="wide"
)

# ---------------- LOAD DATASET ----------------

df = pd.read_csv("Data - Parkinsons.csv")

# ---------------- PREPARE DATA ----------------

X = df.drop(columns=['name', 'status'])
Y = df['status']

# ---------------- TRAIN TEST SPLIT ----------------

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=2
)

# ---------------- SCALING ----------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------- MODEL ----------------

model = LogisticRegression()

model.fit(X_train, Y_train)

# ---------------- SIDEBAR ----------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Home",
        "Disease Prediction",
        "Statistics",
        "About"
    ]
)

# =========================================================
# HOME PAGE
# =========================================================

if page == "Home":

    st.title("🩺 Parkinson Disease Prediction System")

    st.markdown("""
    ## Welcome

    This is a Machine Learning project developed by **Dharshana**
    using **Python** and **Streamlit**.

    ### Features

    - Parkinson Disease Prediction
    - Statistical Analysis
    - Interactive Visualizations
    - Machine Learning Prediction
    - Medical Report Input

    ### Objective

    This system predicts whether a person may have Parkinson Disease
    using medical voice measurement parameters.
    """)

    st.success("Use the sidebar to navigate through the application.")

# =========================================================
# PREDICTION PAGE
# =========================================================

elif page == "Disease Prediction":

    st.title("🧠 Parkinson Disease Prediction")

    st.write("Enter the medical parameters below:")

    feature_names = [
        'MDVP:Fo(Hz)',
        'MDVP:Fhi(Hz)',
        'MDVP:Flo(Hz)',
        'MDVP:Jitter(%)',
        'MDVP:Jitter(Abs)',
        'MDVP:RAP',
        'MDVP:PPQ',
        'Jitter:DDP',
        'MDVP:Shimmer',
        'MDVP:Shimmer(dB)',
        'Shimmer:APQ3',
        'Shimmer:APQ5',
        'MDVP:APQ',
        'Shimmer:DDA',
        'NHR',
        'HNR',
        'RPDE',
        'DFA',
        'spread1',
        'spread2',
        'D2',
        'PPE'
    ]

    features = []

    col1, col2 = st.columns(2)

    for i, feature in enumerate(feature_names):

        with col1 if i % 2 == 0 else col2:

            value = st.number_input(
                feature,
                format="%.6f"
            )

            features.append(value)

    if st.button("Predict Disease"):

        input_data = np.asarray(features)

        input_data = input_data.reshape(1, -1)

        scaled_data = scaler.transform(input_data)

        prediction = model.predict(scaled_data)

        if prediction[0] == 1:

            st.error("⚠️ The person may have Parkinson Disease.")

            st.warning(
                "Please consult a neurologist or doctor for proper diagnosis."
            )

        else:

            st.success(
                "✅ The person does not show Parkinson Disease symptoms."
            )

# =========================================================
# STATISTICS PAGE
# =========================================================

elif page == "Statistics":

    st.title("📊 Statistical Analysis Dashboard")

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    st.subheader("Dataset Shape")

    st.write(df.shape)

    st.subheader("Statistical Summary")

    st.write(df.describe())

    # ---------------- COLUMN SELECTION ----------------

    column = st.selectbox(
        "Select Column",
        df.columns[1:]
    )

    # ---------------- BASIC STATISTICS ----------------

    st.subheader("Basic Statistical Functions")

    col1, col2 = st.columns(2)

    with col1:

        if st.button("Mean"):
            st.write("Mean:", df[column].mean())

        if st.button("Median"):
            st.write("Median:", df[column].median())

    with col2:

        if st.button("Maximum"):
            st.write("Maximum:", df[column].max())

        if st.button("Minimum"):
            st.write("Minimum:", df[column].min())

    # ---------------- VISUALIZATIONS ----------------

    st.subheader("Data Visualization")

    chart = st.selectbox(
        "Select Graph Type",
        [
            "Bar Chart",
            "Pie Chart",
            "Line Chart",
            "Histogram"
        ]
    )

    # BAR CHART

    if chart == "Bar Chart":

        fig = px.bar(
            df,
            x='status',
            y=column,
            title="Bar Chart"
        )

        st.plotly_chart(fig)

    # PIE CHART

    elif chart == "Pie Chart":

        pie = df['status'].value_counts()

        fig = px.pie(
            values=pie.values,
            names=['Parkinson', 'Healthy'],
            title="Pie Chart"
        )

        st.plotly_chart(fig)

    # LINE CHART

    elif chart == "Line Chart":

        fig = px.line(
            df,
            y=column,
            title="Line Chart"
        )

        st.plotly_chart(fig)

    # HISTOGRAM

    elif chart == "Histogram":

        fig = px.histogram(
            df,
            x=column,
            title="Histogram"
        )

        st.plotly_chart(fig)

# =========================================================
# ABOUT PAGE
# =========================================================

elif page == "About":

    st.title("ℹ️ About This Project")

    st.markdown("""
    ## Parkinson Disease Prediction System

    This project was developed by **Dharshana**
    using Python, Machine Learning, and Streamlit.

    ### Technologies Used

    - Python
    - Streamlit
    - Pandas
    - NumPy
    - Scikit-Learn
    - Plotly

    ### Features

    - Disease Prediction
    - Statistical Analysis
    - Interactive Graphs
    - Machine Learning Model

    ### Disclaimer

    This application is developed for educational purposes only.

    Please consult a medical professional for actual diagnosis.
    """)

    st.info("Machine Learning Project using Python and Streamlit")

