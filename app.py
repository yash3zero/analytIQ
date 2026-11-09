
import streamlit as st
import pandas as pd
import plotly.express as px

from pathlib import Path
from src.analyzer import analyze_reviews, generate_summary


# Project directory
BASE_DIR = Path(__file__).resolve().parent

# Streamlit configuration
st.set_page_config(
    page_title="analytIQ",
    page_icon="📊",
    layout="wide"
)


# Custom styling
st.markdown("""
<style>
.stApp {
    background-color: #111318;
}

[data-testid="stMetric"] {
    background: #1b2330;
    padding: 16px;
    border-radius: 12px;
}

.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #ffffff;
}

.subtitle {
    color: #9ca3af;
    font-size: 13px;
    letter-spacing: 2px;
}
</style>
""", unsafe_allow_html=True)


# Sidebar
st.sidebar.title("analytIQ")

st.sidebar.caption(
    "PRODUCT FEEDBACK INTELLIGENCE"
)

st.sidebar.markdown("---")

st.sidebar.subheader("Analysis Settings")

mode = st.sidebar.radio(
    "Input method",
    [
        "Upload CSV",
        "Manual Review",
        "Demo Dataset"
    ]
)


# Input handling
reviews = []

if mode == "Upload CSV":

    uploaded = st.file_uploader(
        "Upload product reviews CSV",
        type=["csv"]
    )

    if uploaded is not None:

        try:
            df = pd.read_csv(uploaded)

            if df.empty:
                st.error("The uploaded CSV is empty.")

            else:

                column = st.selectbox(
                    "Select review text column",
                    df.columns,
                    index=(
                        list(df.columns).index("review_text")
                        if "review_text" in df.columns
                        else 0
                    )
                )

                reviews = (
                    df[column]
                    .dropna()
                    .astype(str)
                    .tolist()
                )

        except Exception as error:
            st.error(f"Unable to read CSV: {error}")


elif mode == "Manual Review":

    manual_text = st.text_area(
        "Enter customer reviews",
        height=220,
        placeholder=(
            "The battery backup is excellent.\n"
            "The camera quality is disappointing.\n"
            "The display is beautiful."
        )
    )

    reviews = [
        line.strip()
        for line in manual_text.splitlines()
        if line.strip()
    ]


elif mode == "Demo Dataset":

    demo_path = BASE_DIR / "data" / "reviews.csv"

    if demo_path.exists():

        demo = pd.read_csv(demo_path)

        reviews = (
            demo["review_text"]
            .dropna()
            .astype(str)
            .tolist()
        )

        st.info(
            "Running analytIQ using the built-in demo dataset."
        )

    else:

        st.error(
            "Demo dataset not found. "
            "Please upload a CSV instead."
        )


# Main dashboard
st.markdown(
    '<div class="main-title">analytIQ</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'ASPECT-BASED E-COMMERCE FEEDBACK ANALYSIS'
    '</div>',
    unsafe_allow_html=True
)


if reviews:

    st.markdown("---")

    st.subheader("Analysis Dashboard")

    with st.spinner("Analyzing customer feedback..."):

        results = analyze_reviews(reviews)

    result_df = pd.DataFrame(results)

    if result_df.empty:

        st.warning(
            "No aspects could be identified in the reviews."
        )

        st.stop()

    # KPI calculations
    total_reviews = len(reviews)

    positive = int(
        (result_df["sentiment"] == "positive").sum()
    )

    negative = int(
        (result_df["sentiment"] == "negative").sum()
    )

    neutral = int(
        (result_df["sentiment"] == "neutral").sum()
    )

    total_aspects = result_df["aspect"].nunique()

    # KPI cards
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Reviews Analyzed",
        total_reviews
    )

    col2.metric(
        "Positive",
        positive
    )

    col3.metric(
        "Negative",
        negative
    )

    col4.metric(
        "Neutral",
        neutral
    )

    col5.metric(
        "Aspects Found",
        total_aspects
    )

    st.caption(
        f"Total aspect-level observations: {len(result_df)}"
    )

    st.markdown("---")

    # Charts
    left, right = st.columns(2)

    with left:

        st.subheader(
            "Aspect-Level Sentiment Distribution"
        )

        sentiment_counts = (
            result_df["sentiment"]
            .value_counts()
            .reindex(
                ["positive", "negative", "neutral"],
                fill_value=0
            )
            .rename_axis("Sentiment")
            .reset_index(name="Count")
        )

        fig = px.pie(
            sentiment_counts,
            names="Sentiment",
            values="Count",
            hole=0.55,
            color="Sentiment",
            color_discrete_map={
                "positive": "#22c55e",
                "negative": "#ef4444",
                "neutral": "#eab308"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        st.subheader("Aspect Distribution")

        aspect_counts = (
            result_df["aspect"]
            .value_counts()
            .rename_axis("Aspect")
            .reset_index(name="Count")
        )

        fig = px.bar(
            aspect_counts,
            x="Aspect",
            y="Count",
            color="Count",
            template="plotly_dark"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    # Aspect analysis
    st.subheader(
        "Aspect-Based Sentiment Analysis"
    )

    aspect_summary = pd.crosstab(
        result_df["aspect"],
        result_df["sentiment"]
    )

    st.dataframe(
        aspect_summary,
        use_container_width=True
    )

    # Aspect sentiment chart
    st.subheader(
        "Aspect Sentiment Breakdown"
    )

    aspect_percentages = (
        result_df
        .groupby(["aspect", "sentiment"])
        .size()
        .reset_index(name="Count")
    )

    fig = px.bar(
        aspect_percentages,
        x="aspect",
        y="Count",
        color="sentiment",
        barmode="group",
        color_discrete_map={
            "positive": "#22c55e",
            "negative": "#ef4444",
            "neutral": "#eab308"
        },
        template="plotly_dark"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    # Summary
    st.subheader(
        "Automated Customer Feedback Summary"
    )

    summary = generate_summary(results)

    st.info(summary)

    st.markdown("---")

    # Detailed results
    st.subheader("Detailed Review Analysis")

    st.dataframe(
        result_df[
            [
                "review_id",
                "aspect",
                "sentiment",
                "confidence",
                "sentence"
            ]
        ],
        use_container_width=True
    )

    # Download results
    csv = result_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Analysis CSV",
        data=csv,
        file_name="analytIQ_analysis.csv",
        mime="text/csv"
    )

else:

    st.markdown("---")

    st.info(
        "Upload a CSV, enter reviews manually, "
        "or select Demo Dataset from the sidebar."
    )

    st.markdown(
        "### Transform customer feedback into product intelligence."
    )


st.markdown("---")

st.caption(
    "analytIQ | Machine Learning Project | 2026"
)
