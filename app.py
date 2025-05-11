import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import tempfile # Import tempfile
# Import gdown
try:
    import gdown
except ImportError:
    st.error("The 'gdown' library is not installed. Please add 'gdown' to your requirements.txt file.")
    st.stop()

import warnings

warnings.filterwarnings("ignore")

# Set page configuration
st.set_page_config(layout="wide", page_title="Beijing Air Quality Analysis and Prediction")

# Loading Data

@st.cache_data # Cache data loading for better performance
def load_data(data_path='merged_beijing_air_quality.csv'):
    """Loads the merged air quality dataset."""
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime']) # Ensure datetime is datetime object
    return df

GOOGLE_DRIVE_FILE_ID = '1LUlsKzS37YGI8AqY71oAj9EJFYM1_9yV'
MODEL_FILENAME = 'model.joblib' 

@st.cache_resource # Cache the model loading
def load_model_from_drive(file_id, output_filename):
    st.write(f"Attempting to download model file (ID: {file_id}) from Google Drive...")
    # Create a temporary directory to store the downloaded file
    temp_dir = tempfile.mkdtemp()
    local_model_path = os.path.join(temp_dir, output_filename)

    try:
        # Use gdown to download the file
        gdown.download(id=file_id, output=local_model_path, quiet=False)
        st.write(f"Model downloaded to {local_model_path}. Loading model...")

        # Load the model from the temporary file
        model = joblib.load(local_model_path)
        st.write("Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error downloading or loading model from Google Drive: {e}")
        st.warning("Please ensure the Google Drive file ID is correct and the file is shared with 'Anyone with the link'.")
        return None
    finally:
        # Clean up the temporary file and directory after loading
        if os.path.exists(local_model_path):
            os.remove(local_model_path)
        if os.path.exists(temp_dir):
             os.rmdir(temp_dir)


# Load data and model
merged_df = load_data()
# Call the new function to load the model from Google Drive
model = load_model_from_drive(GOOGLE_DRIVE_FILE_ID, MODEL_FILENAME)


# Check if data and model loaded successfully
if merged_df is None or model is None:
    st.stop() # Stop the app if essential files are missing

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "Exploratory Data Analysis (EDA)", "Modelling and Prediction"])

# --- Page Content ---

if page == "Data Overview":
    st.title("📊 Data Overview")
    st.write("Information about the Beijing Air Quality Dataset (March 2013 - February 2017).")

    st.subheader("Dataset Head")
    st.dataframe(merged_df.head())

    st.subheader("Dataset Information")
    # Use a string buffer to capture info() output
    from io import StringIO
    buffer = StringIO()
    merged_df.info(buf=buffer)
    st.text(buffer.getvalue())

    st.subheader("Descriptive Statistics")
    st.dataframe(merged_df.describe())

    st.subheader("Unique Value Counts (Categorical Columns)")
    st.write("**Station:**")
    st.write(merged_df['station'].value_counts())
    st.write("**Category:**")
    st.write(merged_df['category'].value_counts())
    st.write("**Wind Direction (wd):**")
    st.write(merged_df['wd'].value_counts())

    st.subheader("Missing Values")
    missing_values = merged_df.isnull().sum()
    missing_percentage = (merged_df.isnull().sum() / len(merged_df)) * 100
    missing_info = pd.DataFrame({'Missing Count': missing_values, 'Missing Percentage (%)': missing_percentage})
    st.dataframe(missing_info[missing_info['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False))
    st.info("Note: Missing numerical values were filled using the forward-fill method, and missing 'wd' values were filled with the mode.")


elif page == "Exploratory Data Analysis (EDA)":
    st.title("🔍 Exploratory Data Analysis (EDA)")
    st.write("Visualizations to understand patterns and relationships in the data.")

    st.subheader("Pollutant Distributions")
    pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Pollutant Distributions Across the Selected Sites", fontsize=18, fontweight='bold')
    axes = axes.flatten() # Flatten the 2x3 array of axes for easy iteration

    for i, col in enumerate(pollutants):
        if col in merged_df.columns:
            sns.histplot(merged_df[col], kde=True, bins=50, ax=axes[i],
                         color=sns.color_palette("dark")[i % len(sns.color_palette("dark"))])
            axes[i].set_title(f'Distribution of {col}', fontsize=14)
            axes[i].set_xlabel(col, fontsize=12)
            axes[i].set_ylabel('Frequency', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to prevent title overlap
    st.pyplot(fig)
    st.write("Most pollutant distributions are strongly right-skewed, indicating frequent lower concentrations with occasional high pollution events.")

    st.subheader("Pollutant Levels by Location Category (Violin Plots)")
    category_order = ['Urban', 'Suburban', 'Rural', 'Industrial']
    fig, axes = plt.subplots(2, 3, figsize=(14, 12))
    fig.suptitle("Pollutant Levels by Location Category (Violin Plots)", fontsize=18, fontweight='bold')
    axes = axes.flatten()

    for i, col in enumerate(pollutants):
        if col in merged_df.columns and 'category' in merged_df.columns:
            sns.violinplot(x='category', y=col, data=merged_df, order=category_order, palette="dark",
                           inner="quartile", cut=0, ax=axes[i])
            axes[i].set_title(f'{col} Levels by Location Category', fontsize=14)
            axes[i].set_xlabel('Location Category', fontsize=12)
            axes[i].set_ylabel(f'{col} (Concentration)', fontsize=12)
            axes[i].tick_params(axis='x', rotation=15)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    st.pyplot(fig)
    st.write("Urban and Industrial areas tend to show higher levels of primary pollutants (NO2, CO). PM2.5 and PM10 are high across all locations. O3 is often higher in Rural and Suburban areas.")


    st.subheader("Monthly Average PM2.5 Levels Over Time")
    fig, ax = plt.subplots(figsize=(15, 8))
    temp_df_for_ts = merged_df.set_index('datetime')
    palette_colors = sns.color_palette("dark", n_colors=len(category_order))
    category_colors = {cat: palette_colors[i] for i, cat in enumerate(category_order)}

    for category_name in category_order:
        if category_name in temp_df_for_ts['category'].unique():
            category_data = temp_df_for_ts[temp_df_for_ts['category'] == category_name]
            # Resample to monthly average, handling potential NaNs after ffill
            monthly_avg_pm25 = category_data['PM2.5'].resample('M').mean().dropna()
            ax.plot(monthly_avg_pm25.index, monthly_avg_pm25.values, label=category_name,
                    marker='.', linestyle='-', color=category_colors.get(category_name))

    ax.set_title('Monthly Average PM2.5 Levels by Location Category', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Average PM2.5 (µg/m³)', fontsize=14)
    ax.legend(title='Location Category')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    st.pyplot(fig)
    st.write("Shows a general downward trend and strong seasonality (peaks in winter, lows in summer) across all locations.")

    st.subheader("PM2.5 Relationships with Meteorological Features")
    # Sample for performance
    sample_df = merged_df.sample(n=min(5000, len(merged_df)), random_state=42)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('PM2.5 Relationships with Meteorological Features', fontsize=18, fontweight='bold')
    axes = axes.flatten()

    # PM2.5 vs Wind Speed
    sns.scatterplot(ax=axes[0], x='WSPM', y='PM2.5', data=sample_df, hue='category',
                    alpha=0.6, s=30, hue_order=category_order, palette="dark")
    axes[0].set_title('PM2.5 vs Wind Speed')
    axes[0].set_xlabel('Wind Speed (m/s)')
    axes[0].set_ylabel('PM2.5 (µg/m³)')
    axes[0].grid(True, linestyle='--')

    # PM2.5 vs Temperature
    sns.scatterplot(ax=axes[1], x='TEMP', y='PM2.5', data=sample_df, hue='category',
                    alpha=0.6, s=30, hue_order=category_order, palette="dark")
    axes[1].set_title('PM2.5 vs Temperature')
    axes[1].set_xlabel('Temperature (°C)')
    axes[1].set_ylabel('PM2.5 (µg/m³)')
    axes[1].grid(True, linestyle='--')

    # PM2.5 vs Dewpoint
    sns.scatterplot(ax=axes[2], x='DEWP', y='PM2.5', data=sample_df, hue='category',
                    alpha=0.6, s=30, hue_order=category_order, palette="dark")
    axes[2].set_title('PM2.5 vs Dewpoint')
    axes[2].set_xlabel('Dewpoint (°C)')
    axes[2].set_ylabel('PM2.5 (µg/m³)')
    axes[2].grid(True, linestyle='--')

    # PM2.5 vs Atmospheric Pressure
    sns.scatterplot(ax=axes[3], x='PRES', y='PM2.5', data=sample_df, hue='category',
                    alpha=0.6, s=30, hue_order=category_order, palette="dark")
    axes[3].set_title('PM2.5 vs Atmospheric Pressure')
    axes[3].set_xlabel('Pressure (hPa)')
    axes[3].set_ylabel('PM2.5 (µg/m³)')
    axes[3].grid(True, linestyle='--')

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    st.pyplot(fig)
    st.write("Shows negative correlation with Wind Speed, some negative correlation with Temperature, positive correlation with Dewpoint, and little correlation with Pressure.")


    st.subheader("PM2.5 Relationships with Other Pollutants")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Scatter Plots of PM2.5 vs Other Pollutants', fontsize=20, fontweight='bold')
    axes = axes.flatten()
    pollutants_other = ['PM10', 'SO2', 'NO2', 'CO', 'O3']

    for i, pollutant in enumerate(pollutants_other):
        sns.scatterplot(ax=axes[i], x=pollutant, y='PM2.5', data=sample_df, hue='category',
                        alpha=0.6, s=30, hue_order=category_order, palette='dark')
        axes[i].set_title(f'PM2.5 vs {pollutant}', fontsize=14)
        axes[i].set_xlabel(pollutant)
        axes[i].set_ylabel('PM2.5 (µg/m³)')
        axes[i].grid(True, linestyle='--')

    # Hide the last unused subplot
    axes[-1].set_visible(False)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    st.pyplot(fig)
    st.write("Strong positive correlation with PM10, moderate positive correlations with CO and NO2. Less clear correlation with SO2 and O3.")


    st.subheader("Correlation Matrix")
    correlation_columns = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
    correlation_matrix = merged_df[correlation_columns].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt=".2f", linewidths=.5, annot_kws={"size":10}, ax=ax)
    ax.set_title('Correlation Matrix of Pollutants and Meteorological Factors', fontsize=16, fontweight='bold')
    # ax.tick_params(axis='x', rotation=45, ha='right')
    # ax.tick_params(axis='y', rotation=0)
    st.pyplot(fig)
    st.write("Confirms intercorrelations among pollutants and negative correlation between pollutants and wind speed.")


elif page == "Modelling and Prediction":
    st.title("🧠 Modelling and Prediction")
    st.write("Predict PM2.5 levels using the trained Random Forest Regression model.")

    if model:
        st.subheader("Model Information")
        st.write("The model used is a Random Forest Regressor.")
        st.write("It was trained on hourly air quality and meteorological data.")
        st.write(f"Model loaded successfully: {type(model)}")

        # Display model performance metrics from training (if available or hardcoded)
        st.subheader("Model Performance (from training)")
        st.write(f"- Test RMSE: Approximately 20.95")
        st.write(f"- Test R-squared: Approximately 0.93")
        st.info("These metrics indicate good performance of the model on unseen data.")


        st.subheader("Make a PM2.5 Prediction")
        st.write("Enter the values for the features below to get a PM2.5 prediction.")

        numerical_features = ['PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM', 'hour']
        categorical_features = ['wd', 'category']

        input_data = {}

        st.markdown("#### Pollutant and Meteorological Inputs")
        cols = st.columns(3)
        input_data['PM10'] = cols[0].number_input('PM10 (µg/m³)', value=50.0, format="%.2f")
        input_data['SO2'] = cols[1].number_input('SO2 (µg/m³)', value=10.0, format="%.2f")
        input_data['NO2'] = cols[2].number_input('NO2 (µg/m³)', value=30.0, format="%.2f")

        cols = st.columns(3)
        input_data['CO'] = cols[0].number_input('CO (µg/m³)', value=0.8, format="%.2f")
        input_data['O3'] = cols[1].number_input('O3 (µg/m³)', value=60.0, format="%.2f")
        input_data['TEMP'] = cols[2].number_input('Temperature (°C)', value=15.0, format="%.2f")

        cols = st.columns(3)
        input_data['PRES'] = cols[0].number_input('Pressure (hPa)', value=1012.0, format="%.2f")
        input_data['DEWP'] = cols[1].number_input('Dewpoint (°C)', value=5.0, format="%.2f")
        input_data['RAIN'] = cols[2].number_input('Rain (mm)', value=0.0, format="%.2f")

        cols = st.columns(2)
        input_data['WSPM'] = cols[0].number_input('Wind Speed (m/s)', value=2.0, format="%.2f")
        input_data['hour'] = cols[1].slider('Hour of the Day', 0, 23, 12)

        st.markdown("#### Location and Wind Direction")
        cols = st.columns(2)
        # Get unique values from the loaded data if possible, otherwise use a default list
        wind_directions = merged_df['wd'].unique().tolist() if merged_df is not None and 'wd' in merged_df.columns else ['E', 'ENE', 'ESE', 'N', 'NE', 'NNE', 'NNW', 'NW', 'S', 'SE', 'SSE', 'SSW', 'SW', 'W', 'WNW', 'WSW']
        input_data['wd'] = cols[0].selectbox('Wind Direction', wind_directions)

        location_categories = merged_df['category'].unique().tolist() if merged_df is not None and 'category' in merged_df.columns else ['Urban', 'Suburban', 'Rural', 'Industrial']
        input_data['category'] = cols[1].selectbox('Location Category', location_categories)


        # Create a DataFrame from input data
        input_df = pd.DataFrame([input_data])


        original_feature_order = numerical_features + categorical_features
        # Reindex input_df to match the original training order
        input_df = input_df[original_feature_order]


        if st.button("Predict PM2.5"):
            try:
                # Make prediction using the loaded pipeline
                prediction = model.predict(input_df)
                st.success(f"Predicted PM2.5 Concentration: **{prediction[0]:.2f} µg/m³**")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.warning("Please ensure all input values are valid.")

    else:
        st.warning("Model not loaded. Cannot perform predictions.")

