import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Parkinson Disease Prediction",
    page_icon="🩺",
    layout="wide"
)

# ---------------- LOAD DATA ----------------

df = pd.read_csv("Data - Parkinsons.csv")

# ---------------- TRAIN MODEL ----------------

X = df.drop(columns=['name', 'status'], axis=1)
Y = df['status']

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=2
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()

model.fit(X_train, Y_train)

# ---------------- SIDEBAR ----------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "Home",
        "Disease Prediction",
        "Statistics",
        "About"
    ]
)

# ---------------- HOME PAGE ----------------

if page == "Home":

    st.title("🩺 Parkinson Disease Prediction System")

    st.image(
        "https://cdn-icons-png.flaticon.com/512/2966/2966486.png",
        width=200
    )

    st.markdown("""
    # Welcome

    This is a Machine Learning project developed by **Dharshana**
    using **Python** and **Streamlit**.

    ## Features

    - Parkinson Disease Prediction
    - Statistical Analysis
    - Interactive Charts
    - Data Visualization
    - Machine Learning Model

    ## Project Objective

    This system predicts whether a person may have Parkinson Disease
    using medical voice measurement parameters.

    """)

    st.success("Use the sidebar to navigate through the application.")

# ---------------- PREDICTION PAGE ----------------

elif page == "Disease Prediction":

    st.title("🧠 Parkinson Disease Prediction")

    st.write("Enter the medical values below:")

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
                "Please consult a neurologist or medical professional for proper diagnosis."
            )

        else:

            st.success(
                "✅ The person does not show Parkinson Disease symptoms."
            )

# ---------------- STATISTICS PAGE ----------------

elif page == "Statistics":

    st.title("📊 Statistical Analysis Dashboard")

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    st.subheader("Dataset Shape")

    st.write(df.shape)

    st.subheader("Statistical Summary")

    st.write(df.describe())

    # Select Column

    column = st.selectbox(
        "Select Column",
        df.columns[1:]
    )

    # Basic Statistics

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

    # Visualization

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

    if chart == "Bar Chart":

        fig = px.bar(
            df,
            x='status',
            y=column,
            title="Bar Chart"
        )

        st.plotly_chart(fig)

    elif chart == "Pie Chart":

        pie = df['status'].value_counts()

        fig = px.pie(
            values=pie.values,
            names=['Parkinson', 'Healthy'],
            title="Pie Chart"
        )

        st.plotly_chart(fig)

    elif chart == "Line Chart":

        fig = px.line(
            df,
            y=column,
            title="Line Chart"
        )

        st.plotly_chart(fig)

    elif chart == "Histogram":

        fig = px.histogram(
            df,
            x=column,
            title="Histogram"
        )

        st.plotly_chart(fig)

# ---------------- ABOUT PAGE ----------------

elif page == "About":

    st.title("ℹ️ About This Project")

    st.markdown("""
    # Parkinson Disease Prediction System

    This project was developed by **Dharshana**
    using **Python**, **Machine Learning**, and **Streamlit**.

    ## Technologies Used

    - Python
    - Streamlit
    - Pandas
    - NumPy
    - Scikit-Learn
    - Plotly

    ## Machine Learning Model

    The model predicts Parkinson Disease using
    medical voice measurement parameters.

    ## Features

    - Disease Prediction
    - Statistical Analysis
    - Data Visualization
    - Interactive Dashboard

    ## Disclaimer

    This application is developed for educational purposes only.

    Please consult a doctor for actual medical diagnosis.
    """)

    st.info("Machine Learning Project using Python and Streamlit")
