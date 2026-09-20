import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Flight Price EDA",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Flight Price EDA & Feature Engineering")
st.markdown(
    "Interactive dashboard based on the Flight Price Prediction EDA notebook."
)

# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file.name.lower().endswith(".xlsx"):
        return pd.read_excel(uploaded_file)
    return pd.read_csv(uploaded_file)


# ============================================================
# FEATURE ENGINEERING
# ============================================================
@st.cache_data
def feature_engineering(data):
    df = data.copy()

    # Date of Journey -> Date, Month, Year
    if "Date_of_Journey" in df.columns:
        date_values = pd.to_datetime(
            df["Date_of_Journey"],
            dayfirst=True,
            errors="coerce"
        )

        df["Date"] = date_values.dt.day
        df["Month"] = date_values.dt.month
        df["Year"] = date_values.dt.year

        df.drop("Date_of_Journey", axis=1, inplace=True)

    # Arrival time -> hour and minute
    if "Arrival_Time" in df.columns:
        arrival_time = (
            df["Arrival_Time"]
            .astype(str)
            .str.extract(r"(\d{1,2}:\d{2})")[0]
        )

        arrival_dt = pd.to_datetime(arrival_time, format="%H:%M", errors="coerce")

        df["Arrival_hour"] = arrival_dt.dt.hour
        df["Arrival_min"] = arrival_dt.dt.minute

        df.drop("Arrival_Time", axis=1, inplace=True)

    # Departure time -> hour and minute
    if "Dep_Time" in df.columns:
        dep_time = (
            df["Dep_Time"]
            .astype(str)
            .str.extract(r"(\d{1,2}:\d{2})")[0]
        )

        dep_dt = pd.to_datetime(dep_time, format="%H:%M", errors="coerce")

        df["Departure_hour"] = dep_dt.dt.hour
        df["Departure_min"] = dep_dt.dt.minute

        df.drop("Dep_Time", axis=1, inplace=True)

    # Total Stops -> numeric
    if "Total_Stops" in df.columns:
        stop_mapping = {
            "non-stop": 0,
            "1 stop": 1,
            "2 stops": 2,
            "3 stops": 3,
            "4 stops": 4
        }

        df["Total_Stops"] = (
            df["Total_Stops"]
            .map(stop_mapping)
            .fillna(1)
        )

    # Route is dropped in the notebook
    if "Route" in df.columns:
        df.drop("Route", axis=1, inplace=True)

    # Duration -> hours and minutes
    if "Duration" in df.columns:
        duration_text = df["Duration"].astype(str)

        hours = pd.to_numeric(
            duration_text.str.extract(r"(\d+)\s*h")[0],
            errors="coerce"
        ).fillna(0)

        minutes = pd.to_numeric(
            duration_text.str.extract(r"(\d+)\s*m")[0],
            errors="coerce"
        ).fillna(0)

        df["Duration_hour"] = hours
        df["Duration_min"] = minutes
        df["Duration_total_minutes"] = hours * 60 + minutes
        df["Duration_total_hours"] = df["Duration_total_minutes"] / 60

    return df


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("📂 Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload flight_price.xlsx or CSV",
    type=["xlsx", "xls", "csv"]
)

if uploaded_file is None:
    st.info(
        "👈 Upload the **flight_price.xlsx** dataset from the sidebar to start."
    )
    st.stop()

raw_df = load_data(uploaded_file)
df = feature_engineering(raw_df)

st.sidebar.success(f"{len(df):,} rows loaded")

# ============================================================
# FILTERS
# ============================================================
st.sidebar.header("🔎 Filters")

filtered_df = df.copy()

if "Airline" in df.columns:
    airlines = sorted(df["Airline"].dropna().unique())
    selected_airlines = st.sidebar.multiselect(
        "Airline",
        airlines,
        default=airlines
    )
    filtered_df = filtered_df[
        filtered_df["Airline"].isin(selected_airlines)
    ]

if "Source" in df.columns:
    sources = sorted(df["Source"].dropna().unique())
    selected_sources = st.sidebar.multiselect(
        "Source",
        sources,
        default=sources
    )
    filtered_df = filtered_df[
        filtered_df["Source"].isin(selected_sources)
    ]

if "Destination" in df.columns:
    destinations = sorted(df["Destination"].dropna().unique())
    selected_destinations = st.sidebar.multiselect(
        "Destination",
        destinations,
        default=destinations
    )
    filtered_df = filtered_df[
        filtered_df["Destination"].isin(selected_destinations)
    ]

if "Class" in df.columns:
    classes = sorted(df["Class"].dropna().unique())
    selected_classes = st.sidebar.multiselect(
        "Class",
        classes,
        default=classes
    )
    filtered_df = filtered_df[
        filtered_df["Class"].isin(selected_classes)
    ]

if "Total_Stops" in df.columns:
    stop_values = sorted(df["Total_Stops"].dropna().unique())
    selected_stops = st.sidebar.multiselect(
        "Stops",
        stop_values,
        default=stop_values
    )
    filtered_df = filtered_df[
        filtered_df["Total_Stops"].isin(selected_stops)
    ]

if "Price" in df.columns:
    min_price = float(df["Price"].min())
    max_price = float(df["Price"].max())

    price_range = st.sidebar.slider(
        "Price Range",
        min_value=min_price,
        max_value=max_price,
        value=(min_price, max_price)
    )

    filtered_df = filtered_df[
        filtered_df["Price"].between(
            price_range[0],
            price_range[1]
        )
    ]

# ============================================================
# KPI CARDS
# ============================================================
st.subheader("📊 Flight Price Overview")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Flights",
    f"{len(filtered_df):,}"
)

c2.metric(
    "Average Price",
    f"₹{filtered_df['Price'].mean():,.0f}"
    if "Price" in filtered_df.columns else "N/A"
)

c3.metric(
    "Minimum Price",
    f"₹{filtered_df['Price'].min():,.0f}"
    if "Price" in filtered_df.columns else "N/A"
)

c4.metric(
    "Maximum Price",
    f"₹{filtered_df['Price'].max():,.0f}"
    if "Price" in filtered_df.columns else "N/A"
)

c5.metric(
    "Airlines",
    f"{filtered_df['Airline'].nunique():,}"
    if "Airline" in filtered_df.columns else "N/A"
)

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧹 Data & Cleaning",
    "📈 EDA",
    "⏱️ Time & Duration",
    "💰 Price Analysis",
    "🔢 Feature Engineering"
])

# ============================================================
# TAB 1 - DATA
# ============================================================
with tab1:
    st.subheader("Raw Dataset")

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", f"{len(raw_df):,}")
    col2.metric("Columns", f"{len(raw_df.columns):,}")
    col3.metric("Duplicate Rows", f"{raw_df.duplicated().sum():,}")

    st.dataframe(
        raw_df.head(20),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Dataset Information")

    info_df = pd.DataFrame({
        "Column": raw_df.columns,
        "Data Type": raw_df.dtypes.astype(str).values,
        "Missing Values": raw_df.isnull().sum().values,
        "Unique Values": raw_df.nunique().values
    })

    st.dataframe(
        info_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Descriptive Statistics")

    st.dataframe(
        raw_df.describe(include="all").T,
        use_container_width=True
    )

    st.subheader("Missing Values")

    missing_df = (
        raw_df.isnull()
        .sum()
        .reset_index()
    )

    missing_df.columns = ["Column", "Missing Values"]
    missing_df = missing_df[
        missing_df["Missing Values"] > 0
    ].sort_values("Missing Values", ascending=False)

    if missing_df.empty:
        st.success("No missing values found.")
    else:
        fig = px.bar(
            missing_df,
            x="Missing Values",
            y="Column",
            orientation="h",
            title="Missing Values by Column"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature-Engineered Dataset")

    st.dataframe(
        df.head(20),
        use_container_width=True,
        hide_index=True
    )

    cleaned_csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Feature-Engineered CSV",
        data=cleaned_csv,
        file_name="flight_price_feature_engineered.csv",
        mime="text/csv"
    )

# ============================================================
# TAB 2 - EDA
# ============================================================
with tab2:
    st.subheader("Airline Distribution")

    airline_counts = (
        filtered_df["Airline"]
        .value_counts()
        .reset_index()
    )
    airline_counts.columns = ["Airline", "Flights"]

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            airline_counts,
            x="Airline",
            y="Flights",
            title="Number of Flights by Airline"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.pie(
            airline_counts,
            names="Airline",
            values="Flights",
            title="Airline Share"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Source and Destination")

    col1, col2 = st.columns(2)

    with col1:
        source_counts = (
            filtered_df["Source"]
            .value_counts()
            .reset_index()
        )
        source_counts.columns = ["Source", "Flights"]

        fig = px.bar(
            source_counts,
            x="Source",
            y="Flights",
            title="Flights by Source City"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        destination_counts = (
            filtered_df["Destination"]
            .value_counts()
            .reset_index()
        )
        destination_counts.columns = ["Destination", "Flights"]

        fig = px.bar(
            destination_counts,
            x="Destination",
            y="Flights",
            title="Flights by Destination City"
        )
        st.plotly_chart(fig, use_container_width=True)

    if "Class" in filtered_df.columns:
        st.subheader("Class Distribution")

        class_counts = (
            filtered_df["Class"]
            .value_counts()
            .reset_index()
        )
        class_counts.columns = ["Class", "Flights"]

        fig = px.pie(
            class_counts,
            names="Class",
            values="Flights",
            title="Economy vs Business"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("The uploaded dataset does not contain a 'Class' column, so Class Distribution is skipped.")

    st.subheader("Number of Stops")

    stop_counts = (
        filtered_df["Total_Stops"]
        .value_counts()
        .sort_index()
        .reset_index()
    )
    stop_counts.columns = ["Stops", "Flights"]

    fig = px.bar(
        stop_counts,
        x="Stops",
        y="Flights",
        title="Flights by Number of Stops"
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 3 - TIME & DURATION
# ============================================================
with tab3:
    st.subheader("Departure Time Analysis")

    if "Departure_hour" in filtered_df.columns:
        dep_hour = (
            filtered_df["Departure_hour"]
            .value_counts()
            .sort_index()
            .reset_index()
        )
        dep_hour.columns = ["Hour", "Flights"]

        fig = px.line(
            dep_hour,
            x="Hour",
            y="Flights",
            markers=True,
            title="Flights by Departure Hour"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Arrival Time Analysis")

    if "Arrival_hour" in filtered_df.columns:
        arr_hour = (
            filtered_df["Arrival_hour"]
            .value_counts()
            .sort_index()
            .reset_index()
        )
        arr_hour.columns = ["Hour", "Flights"]

        fig = px.line(
            arr_hour,
            x="Hour",
            y="Flights",
            markers=True,
            title="Flights by Arrival Hour"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Flight Duration")

    if "Duration_total_hours" in filtered_df.columns:
        fig = px.histogram(
            filtered_df,
            x="Duration_total_hours",
            nbins=40,
            marginal="box",
            title="Flight Duration Distribution",
            labels={
                "Duration_total_hours": "Duration (Hours)"
            }
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Duration vs Price")

    if "Duration_total_hours" in filtered_df.columns:
        fig = px.scatter(
            filtered_df,
            x="Duration_total_hours",
            y="Price",
            color="Class" if "Class" in filtered_df.columns else None,
            hover_data=["Airline", "Source", "Destination"],
            title="Flight Duration vs Ticket Price"
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 4 - PRICE ANALYSIS
# ============================================================
with tab4:
    st.subheader("Average Price by Airline")

    avg_airline_price = (
        filtered_df.groupby("Airline", as_index=False)["Price"]
        .mean()
        .sort_values("Price", ascending=False)
    )

    fig = px.bar(
        avg_airline_price,
        x="Airline",
        y="Price",
        title="Average Ticket Price by Airline",
        labels={"Price": "Average Price"}
    )
    st.plotly_chart(fig, use_container_width=True)

    if "Class" in filtered_df.columns:
        st.subheader("Price by Class")

        class_price = (
            filtered_df.groupby("Class", as_index=False)["Price"]
            .mean()
        )

        fig = px.bar(
            class_price,
            x="Class",
            y="Price",
            title="Average Price by Class"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("The uploaded dataset does not contain a 'Class' column, so Price by Class is skipped.")

    st.subheader("Price by Number of Stops")

    stop_price = (
        filtered_df.groupby("Total_Stops", as_index=False)["Price"]
        .mean()
    )

    fig = px.bar(
        stop_price,
        x="Total_Stops",
        y="Price",
        title="Average Price by Number of Stops"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Price Distribution")

    fig = px.histogram(
        filtered_df,
        x="Price",
        nbins=50,
        marginal="box",
        title="Ticket Price Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Price by Source and Destination")

    route_price = (
        filtered_df
        .groupby(["Source", "Destination"], as_index=False)["Price"]
        .mean()
    )

    fig = px.density_heatmap(
        route_price,
        x="Source",
        y="Destination",
        z="Price",
        histfunc="avg",
        text_auto=".0f",
        title="Average Price: Source vs Destination"
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 5 - FEATURE ENGINEERING
# ============================================================
with tab5:
    st.subheader("Features Created from the Notebook")

    feature_description = pd.DataFrame({
        "Feature": [
            "Date",
            "Month",
            "Year",
            "Arrival_hour",
            "Arrival_min",
            "Departure_hour",
            "Departure_min",
            "Total_Stops",
            "Duration_hour",
            "Duration_min",
            "Duration_total_minutes",
            "Duration_total_hours"
        ],
        "Description": [
            "Day extracted from Date_of_Journey",
            "Month extracted from Date_of_Journey",
            "Year extracted from Date_of_Journey",
            "Hour extracted from Arrival_Time",
            "Minute extracted from Arrival_Time",
            "Hour extracted from Dep_Time",
            "Minute extracted from Dep_Time",
            "Stops converted to numeric values",
            "Hours extracted from Duration",
            "Minutes extracted from Duration",
            "Total journey duration in minutes",
            "Total journey duration in hours"
        ]
    })

    st.dataframe(
        feature_description,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Numeric Correlation")

    numeric_columns = [
        col for col in [
            "Price",
            "Date",
            "Month",
            "Year",
            "Arrival_hour",
            "Arrival_min",
            "Departure_hour",
            "Departure_min",
            "Total_Stops",
            "Duration_hour",
            "Duration_min",
            "Duration_total_minutes",
            "Duration_total_hours"
        ]
        if col in filtered_df.columns
    ]

    if len(numeric_columns) >= 2:
        correlation = filtered_df[numeric_columns].corr()

        fig = px.imshow(
            correlation,
            text_auto=".2f",
            aspect="auto",
            title="Correlation Matrix"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("One-Hot Encoding Preview")

    categorical_columns = [
        col for col in ["Airline", "Source", "Destination"]
        if col in filtered_df.columns
    ]

    if categorical_columns:
        encoding_preview = pd.get_dummies(
            filtered_df[categorical_columns],
            columns=categorical_columns,
            dtype=int
        )

        st.write(
            f"Original categorical columns: "
            f"**{', '.join(categorical_columns)}**"
        )

        st.write(
            f"Encoded feature count: **{encoding_preview.shape[1]}**"
        )

        st.dataframe(
            encoding_preview.head(10),
            use_container_width=True,
            hide_index=True
        )

        encoded_csv = encoding_preview.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download One-Hot Encoded Features",
            data=encoded_csv,
            file_name="flight_price_encoded_features.csv",
            mime="text/csv"
        )

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption(
    "Flight Price EDA Dashboard | Feature Engineering | Interactive Analysis"
)
