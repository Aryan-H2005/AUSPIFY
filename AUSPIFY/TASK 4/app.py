import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# =========================================================
# 1. PAGE SETUP
# =========================================================

st.set_page_config(
    page_title="Netflix Data Science Dashboard",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Netflix Data Science Dashboard")
st.write("Data Analysis, Machine Learning and Business Insights")


# =========================================================
# 2. LOAD DATASET
# =========================================================

df = pd.read_csv("clean_dataset.csv")

st.success("Dataset loaded successfully!")


# =========================================================
# 3. SIDEBAR
# =========================================================

st.sidebar.title("Dashboard")

option = st.sidebar.selectbox(
    "Select Section",
    [
        "Overview",
        "Content Analysis",
        "Country Analysis",
        "Category Analysis",
        "Release Year Analysis",
        "Machine Learning",
        "Business Insights"
    ]
)


# =========================================================
# 4. OVERVIEW
# =========================================================

if option == "Overview":

    st.header("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Titles",
        len(df)
    )

    col2.metric(
        "Movies",
        len(df[df["type"] == "Movie"])
    )

    col3.metric(
        "TV Shows",
        len(df[df["type"] == "TV Show"])
    )

    st.subheader("Dataset Preview")

    st.dataframe(df.head(20))

    st.subheader("Dataset Information")

    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])

    st.subheader("Columns")

    st.write(list(df.columns))


# =========================================================
# 5. CONTENT ANALYSIS
# =========================================================

elif option == "Content Analysis":

    st.header("Movie vs TV Show Analysis")

    type_count = df["type"].value_counts()

    st.write(type_count)

    fig, ax = plt.subplots()

    ax.bar(
        type_count.index,
        type_count.values
    )

    ax.set_title("Movies vs TV Shows")
    ax.set_xlabel("Content Type")
    ax.set_ylabel("Number of Titles")

    st.pyplot(fig)


# =========================================================
# 6. COUNTRY ANALYSIS
# =========================================================

elif option == "Country Analysis":

    st.header("Top Countries")

    country_count = (
        df["country"]
        .dropna()
        .value_counts()
        .head(10)
    )

    st.bar_chart(country_count)

    st.subheader("Top 10 Countries")

    st.dataframe(
        country_count.reset_index()
    )


# =========================================================
# 7. CATEGORY ANALYSIS
# =========================================================

elif option == "Category Analysis":

    st.header("Top Netflix Categories")

    category_count = (
        df["listed_in"]
        .dropna()
        .value_counts()
        .head(10)
    )

    st.bar_chart(category_count)

    st.subheader("Top 10 Categories")

    st.dataframe(
        category_count.reset_index()
    )


# =========================================================
# 8. RELEASE YEAR ANALYSIS
# =========================================================

elif option == "Release Year Analysis":

    st.header("Netflix Content by Release Year")

    year_count = (
        df["release_year"]
        .value_counts()
        .sort_index()
    )

    st.line_chart(year_count)

    st.subheader("Release Year Data")

    st.dataframe(
        year_count.reset_index()
    )


# =========================================================
# 9. MACHINE LEARNING
# =========================================================

elif option == "Machine Learning":

    st.header("🎯 Netflix Content Classification")

    st.write(
        "The model predicts whether a title is a Movie or TV Show."
    )

    # Select required columns
    ml_df = df[
        [
            "type",
            "release_year",
            "rating"
        ]
    ].dropna()

    # Encode rating
    rating_encoder = LabelEncoder()

    ml_df["rating"] = rating_encoder.fit_transform(
        ml_df["rating"]
    )

    # Encode target
    type_encoder = LabelEncoder()

    ml_df["type"] = type_encoder.fit_transform(
        ml_df["type"]
    )

    # Features
    X = ml_df[
        [
            "release_year",
            "rating"
        ]
    ]

    # Target
    y = ml_df["type"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Create model
    model = DecisionTreeClassifier(
        random_state=42
    )

    # Train
    model.fit(
        X_train,
        y_train
    )

    # Predict
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    st.metric(
        "Model Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    st.write(
        "The model was trained using release year and rating."
    )

    st.subheader("Make a Prediction")

    release_year = st.number_input(
        "Release Year",
        min_value=1900,
        max_value=2030,
        value=2020
    )

    rating_list = list(
        rating_encoder.classes_
    )

    rating = st.selectbox(
        "Rating",
        rating_list
    )

    if st.button("Predict"):

        rating_number = rating_encoder.transform(
            [rating]
        )[0]

        prediction = model.predict(
            [[
                release_year,
                rating_number
            ]]
        )

        result = type_encoder.inverse_transform(
            prediction
        )[0]

        st.success(
            f"Predicted Content Type: {result}"
        )


# =========================================================
# 10. BUSINESS INSIGHTS
# =========================================================

elif option == "Business Insights":

    st.header("💡 Business Insights")

    movie_count = len(
        df[df["type"] == "Movie"]
    )

    tv_count = len(
        df[df["type"] == "TV Show"]
    )

    top_country = (
        df["country"]
        .dropna()
        .value_counts()
        .index[0]
    )

    top_category = (
        df["listed_in"]
        .dropna()
        .value_counts()
        .index[0]
    )

    st.subheader("Key Findings")

    st.write(
        f"• The dataset contains {len(df)} Netflix titles."
    )

    st.write(
        f"• There are {movie_count} Movies and {tv_count} TV Shows."
    )

    st.write(
        f"• {top_country} is one of the leading countries "
        "by number of titles in the dataset."
    )

    st.write(
        f"• The most common category in the dataset is "
        f"{top_category}."
    )

    st.write(
        "• Release-year analysis can help understand "
        "content growth and changing trends."
    )

    st.write(
        "• Category and country analysis can support "
        "content planning and regional strategies."
    )

    st.subheader("Conclusion")

    st.write(
        "The dashboard combines data analysis, "
        "visualization, machine learning and business "
        "insights to understand the Netflix content library."
    )


# =========================================================
# 11. FOOTER
# =========================================================

st.sidebar.markdown("---")
st.sidebar.write("Netflix Data Science Project")
st.sidebar.write("Python • Pandas • Matplotlib • Scikit-learn • Streamlit")