import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --- App Configuration ---
st.set_page_config(page_title="DataWiz Pro", layout="wide", initial_sidebar_state="expanded")

# --- Initialize Session State ---
if "df" not in st.session_state:
    st.session_state.df = None
if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None
if "step" not in st.session_state:
    st.session_state.step = "Upload Data"

# --- App Header ---
st.title("🚀 DataWiz Pro")
st.markdown("**Explore, Clean, Visualize, and Predict Data Like a Pro!**")

# --- Sidebar Navigation ---
st.sidebar.markdown("## 🚀 **DataWiz Pro**")
st.sidebar.markdown("Explore, clean, visualize, and predict data seamlessly! 🌟")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📂 **Navigation**")
sections = {
    "🏠 Home": "Home",
    "📤 Upload Data": "Upload Data",
    "🛠️ Data Cleaning": "Data Cleaning",
    "🔄 Data Transformation": "Data Transformation",
    "📊 Data Visualization": "Data Visualization",
    "📈 Model Prediction": "Model Prediction",
    "ℹ️ About": "About"
}

# Use icons and a cleaner layout
choice = st.sidebar.radio(
    "Choose a Section:",
    options=list(sections.keys()),
    format_func=lambda x: x,
)

# Map the choice to the corresponding section
section = sections[choice]

# --- Home Section ---
if choice == "🏠 Home":
    st.header("Welcome to the DataWiz Pro!")
    st.write(
        """
        This app offers the following features:
        - Upload datasets for exploration.
        - Clean and preprocess data with advanced tools.
        - Visualize data with customizable, interactive plots.
        - Transform data and add new calculated fields.
        - Predict outcomes using a simple linear regression model.
        """
    )
    st.image("https://via.placeholder.com/800x400?text=DataWiz+Pro", use_container_width=True)


# --- Upload Data Section ---
elif choice == "📤 Upload Data":
    st.header("Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a file (CSV or Excel)", type=["csv", "xlsx"])

    if uploaded_file:
        # Load the dataset
        if uploaded_file.name.endswith(".csv"):
            st.session_state.df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            st.session_state.df = pd.read_excel(uploaded_file)

        st.write("### Raw Dataset")
        st.dataframe(st.session_state.df)

        # Automatically clean the dataset
        df = st.session_state.df.copy()

        # Auto-cleaning steps
        # Handle Missing Values (Fill with mean for numeric, mode for categorical)
        for column in df.columns:
            if df[column].isnull().sum() > 0:
                if df[column].dtype in ["float64", "int64"]:
                    df[column].fillna(df[column].mean(), inplace=True)
                else:
                    df[column].fillna(df[column].mode()[0], inplace=True)

        # Remove Duplicates
        df.drop_duplicates(inplace=True)

        # Standardize Column Names
        df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

        # Save cleaned data to session state
        st.session_state.cleaned_df = df

        # Prompt user to go to Data Cleaning section
        st.success("Dataset uploaded successfully! Please go to the **Data Cleaning** section to review your cleaned data.")
    else:
        st.info("Please upload a dataset to start.")

# --- Data Cleaning Section ---
elif choice == "🛠️ Data Cleaning":
    st.header("Cleaned Dataset")
    if st.session_state.cleaned_df is not None:
        st.markdown("#### Cleaning Steps Applied Automatically:")
        st.markdown(
            """
            - **Missing Values**: Filled numeric columns with mean, and categorical columns with mode.
            - **Duplicates**: Removed duplicate rows.
            - **Column Names**: Standardized to lowercase and replaced spaces with underscores.
            """
        )
        st.write("### Cleaned Dataset")
        st.dataframe(st.session_state.cleaned_df)

        # Option to download cleaned dataset
        st.download_button(
            label="Download Cleaned Dataset",
            data=st.session_state.cleaned_df.to_csv(index=False),
            file_name="cleaned_data.csv",
            mime="text/csv",
        )
    else:
        st.warning("Please upload a dataset in the **Upload Data** section first.")

# --- Data Transformation Section ---
elif choice == "🔄 Data Transformation":
    st.header("Data Transformation Tools")
    
    if st.session_state.cleaned_df is not None:
        df = st.session_state.cleaned_df.copy()

        # Add Calculated Column
        st.subheader("Add Calculated Column")
        new_col_name = st.text_input("Enter new column name")
        
        if new_col_name:
            formula = st.text_input("Enter calculation formula (e.g., col1 + col2)")
            if formula:
                try:
                    df[new_col_name] = df.eval(formula)
                    st.success(f"Added column: {new_col_name}")
                except Exception as e:
                    st.error(f"Error in calculation: {e}")

        # Filter Rows
        st.subheader("Filter Rows")
        filter_column = st.selectbox("Select Column to Filter", df.columns)

        if filter_column:
            filter_value = st.text_input("Enter filter condition (e.g., > 50, == 'value')")

            if filter_value:
                try:
                    # Check if the filter value is numeric
                    if filter_value.replace('.', '', 1).isdigit():  # Check if value is numeric
                        filter_value = float(filter_value)  # Convert to float for numeric comparison
                        query_str = f"`{filter_column}` {filter_value}"
                    else:
                        # If filter value is a string, ensure it is wrapped in quotes for comparison
                        filter_value = f"'{filter_value}'"
                        query_str = f"`{filter_column}` == {filter_value}"  # Use double equals for comparison

                    # Apply filter using the query string
                    df = df.query(query_str)
                    st.success(f"Applied filter: {filter_column} {filter_value}")

                except Exception as e:
                    st.error(f"Error in filtering: {e}")


        # Display Transformed Data
        st.write("### Transformed Dataset")
        st.dataframe(df)

        # Option to download the transformed dataset
        st.download_button(
            label="Download Transformed Dataset",
            data=df.to_csv(index=False),
            file_name="transformed_data.csv",
            mime="text/csv",
        )

    else:
        st.warning("Please upload and clean a dataset first.")

# --- Data Visualization Section ---
elif choice == "📊 Data Visualization":
    st.header("Data Visualization")
    if st.session_state.cleaned_df is not None:
        df = st.session_state.cleaned_df.copy()
        columns = df.columns

        # Select Plot Type
        plot_type = st.selectbox("Choose Plot Type", ["Histogram", "Box Plot", "Scatter Plot", "Pie Chart", "Heatmap"])
        if plot_type == "Histogram":
            column = st.selectbox("Select Column for Histogram", columns)
            st.plotly_chart(px.histogram(df, x=column))
        elif plot_type == "Box Plot":
            column = st.selectbox("Select Column for Box Plot", columns)
            st.plotly_chart(px.box(df, y=column))
        elif plot_type == "Scatter Plot":
            x_col = st.selectbox("X-Axis", columns)
            y_col = st.selectbox("Y-Axis", columns)
            st.plotly_chart(px.scatter(df, x=x_col, y=y_col))
        elif plot_type == "Pie Chart":
            column = st.selectbox("Select Column for Pie Chart", columns)
            st.plotly_chart(px.pie(df, names=column))
        elif plot_type == "Heatmap":
            st.write("Correlation Heatmap")
            sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
            st.pyplot()
    else:
        st.warning("Please upload and clean a dataset first.")

# --- Model Prediction Section ---
elif choice == "📈 Model Prediction":
    st.header("Linear Regression Prediction")
    if st.session_state.cleaned_df is not None:
        df = st.session_state.cleaned_df.copy()
        numeric_cols = df.select_dtypes(include=["int", "float"]).columns

        # Select Features and Target
        st.subheader("Select Features and Target")
        features = st.multiselect("Choose Features (X)", numeric_cols)
        target = st.selectbox("Choose Target (Y)", numeric_cols)

        if features and target:
            X = df[features]
            y = df[target]

            # Train-Test Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train Model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predict
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)

            st.write(f"### Model Performance")
            st.write(f"Mean Squared Error: {mse:.2f}")
            st.write("### Predictions vs Actuals")
            results = pd.DataFrame({"Actual": y_test, "Predicted": predictions})
            st.dataframe(results)
    else:
        st.warning("Please upload and clean a dataset first.")

# --- About Section ---
elif choice == "ℹ️ About":
    st.header("About")
    st.write("DataWiz Pro is developed by Mohsin Irfan, a passionate data enthusiast dedicated to bringing powerful data analysis and machine learning tools to users. 🚀")
