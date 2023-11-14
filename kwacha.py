import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import plotly.graph_objects as go
from PIL import Image

def read_data(file_path):
    df = pd.read_csv(file_path)
    return df

def create_lag_column(df, lag_days=5):
    df['SELLING RATE (5 Days Lag)'] = df['SELLING RATE'].shift(lag_days)
    df = df.dropna()
    return df

def train_models(X, y):
    lr_model = LinearRegression()
    lr_model.fit(X, y)

    rf_model = RandomForestRegressor()
    rf_model.fit(X, y.ravel())

    dt_model = DecisionTreeRegressor()
    dt_model.fit(X, y.ravel())

    xgb_model = XGBRegressor()
    xgb_model.fit(X, y.ravel())

    return lr_model, rf_model, dt_model, xgb_model

def visualize_data(df):
    fig = go.Figure()

    # Mid Rate
    fig.add_trace(go.Scatter(x=df['DATE'], y=df['MID RATE'], mode='lines', name='Mid Rate'))

    return fig

def generate_stats_output(df):
    stats_df = df.describe()
    return stats_df

def predict_future_values(model, lag_values, days):
    predictions = [model.predict([[lag_value]])[0] for lag_value in lag_values]
    return predictions

def plot_predictions(last_n_days, predictions_df, days_to_predict):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=last_n_days['DATE'], y=last_n_days['SELLING RATE'], mode='lines', name='Selling Rate (Actual)'))

    for model_name, predictions in predictions_df.items():
        fig.add_trace(go.Scatter(x=pd.date_range(last_n_days['DATE'].iloc[-1], periods=days_to_predict + 1).shift(1),
                                 y=[last_n_days['SELLING RATE'].iloc[-1]] + predictions.tolist(),
                                 mode='lines', name=f'{model_name} (Predicted)', line=dict(dash='dash')))

    return fig

def calculate_moving_averages(df):
    df['15 Day MA'] = df['MID RATE'].rolling(window=15).mean()
    df['30 Day MA'] = df['MID RATE'].rolling(window=30).mean()
    df['60 Day MA'] = df['MID RATE'].rolling(window=60).mean()
    return df

def visualize_moving_averages(df):
    fig = go.Figure()

    # Mid Rate
    fig.add_trace(go.Scatter(x=df['DATE'], y=df['MID RATE'], mode='lines', name='Mid Rate'))
    
    # 15 Day Moving Average
    fig.add_trace(go.Scatter(x=df['DATE'], y=df['15 Day MA'], mode='lines', name='15 Day MA'))

    # 30 Day Moving Average
    fig.add_trace(go.Scatter(x=df['DATE'], y=df['30 Day MA'], mode='lines', name='30 Day MA'))

    # 60 Day Moving Average
    fig.add_trace(go.Scatter(x=df['DATE'], y=df['60 Day MA'], mode='lines', name='60 Day MA'))

    return fig

def calculate_volatility(df, window):
    df['Mid Rate Returns'] = df['MID RATE'].pct_change()
    df['Volatility'] = df['Mid Rate Returns'].rolling(window=window).std()
    df = df.dropna()
    return df

def plot_volatility(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['DATE'], y=df['Volatility'], mode='lines', name='Volatility'))
    fig.update_layout(title='Mid Rate Volatility Over Time', xaxis_title='Date', yaxis_title='Volatility')
    return fig

def home_page():
    st.title("Kwacha Forex App")
    st.write("Welcome to the Kwacha Forex App! Explore different features and functionalities.")
    st.divider()  # 👈 Draws a horizontal rul
    image = Image.open('trade.jpg')

    st.image(image, caption='Dollar and Kwacha')
    st.divider()  # 👈 Draws a horizontal rul
    st.markdown(
    """
    Kwacha Forex app is an open-source demo app built using
    Machine Learning and Data Science.

    **👈 Select a Page from the sidebar** to see some features
    of what Kwacha Forex App can do!
    ### Want to learn more?
    - Check out creator Kampamba Shula on Github [Kshula](https://github.com/kshula)
    - email : kampambashula@gmail.com
    """)
    st.divider()  # 👈 Draws a horizontal rul

def about_page():
    st.title("About")
    image = Image.open('future.jpg')

    st.image(image, caption='forex')
    
    st.write("This Forex App provides visualizations, statistical outputs, and predictions based on different models.")
    st.write("Explore the 'Home' page for a brief overview.")
    st.write("Check the 'Visualization' page to see trends in Mid rates with moving averages and volatility.")
    st.write("Visit the 'Stats Output' page for statistical summaries of the data.")
    st.write("Make predictions on the 'Predictions' page using various machine learning models.")
    st.write("Feel free to navigate through different features using the sidebar.")
    
def main():
    file_path = 'data.csv'
    df = read_data(file_path)
    df = create_lag_column(df)
    df = calculate_moving_averages(df)

    # Set the page configuration
    st.set_page_config(page_title="Forex App", layout="wide")

    # Sidebar Navigation
    selected_page = st.sidebar.radio("Navigate", ["Home", "About", "Visualization", "Stats Output", "Predictions"])

    if selected_page == "Home":
        home_page()
    elif selected_page == "About":
        about_page()
    elif selected_page == "Visualization":
        st.title("Visualization")
        st.write("Explore visualizations of the Forex data here.")

        # Calculate and plot moving averages
        df_with_averages = calculate_moving_averages(df.copy())
        fig_averages = visualize_moving_averages(df_with_averages)
        st.plotly_chart(fig_averages)

        # Calculate and plot volatility
        volatility_window = st.sidebar.slider("Select Volatility Window (Days)", 1, 30, 20)
        df_with_volatility = calculate_volatility(df.copy(), volatility_window)
        fig_volatility = plot_volatility(df_with_volatility)
        st.plotly_chart(fig_volatility)

        st.write("Go to the 'Stats Output' or 'Predictions' pages for more analysis.")
    elif selected_page == "Stats Output":
        st.title("Stats Output")
        st.write("View statistical outputs of the Forex data here.")

        # Example stats output using pandas
        stats_df = df.describe()
        st.write(stats_df)

        st.write("Go to the 'Visualization' or 'Predictions' pages for more analysis.")
    elif selected_page == "Predictions":
        st.title("Predictions")
        st.write("Make predictions using different models here.")
        st.write("Adjust slider to predict selling rate for different days")
        st.divider()  # 👈 Draws a horizontal rul
        days_to_predict = st.sidebar.slider("Select Days for Prediction", 1, 15, 5, key="days_to_predict")

        X = df['SELLING RATE (5 Days Lag)'].values.reshape(-1, 1)
        y = df['SELLING RATE'].values.reshape(-1, 1)

        lr_model, rf_model, dt_model, xgb_model = train_models(X, y)

        lag_values = df['SELLING RATE (5 Days Lag)'].values[-days_to_predict:]
        lr_predictions = predict_future_values(lr_model, lag_values, days_to_predict)
        rf_predictions = predict_future_values(rf_model, lag_values, days_to_predict)
        dt_predictions = predict_future_values(dt_model, lag_values, days_to_predict)
        xgb_predictions = predict_future_values(xgb_model, lag_values, days_to_predict)

        predictions_df = pd.DataFrame({
            'Linear Regression': lr_predictions,
            'Random Forest': rf_predictions,
            'Decision Tree': dt_predictions,
            'XGBoost': xgb_predictions
        })

        last_90_days = df.iloc[-90:]
        fig = plot_predictions(last_90_days, predictions_df, days_to_predict)
        st.plotly_chart(fig)

        st.write(f"Predicted Selling Rate for the Next {days_to_predict} Days:")
        st.write(predictions_df)
        st.divider()  # 👈 Draws a horizontal rul
    st.write("Go to the 'About', 'Home', 'Visualization', 'Stats Output', or 'Predictions' pages for more analysis.")

if __name__ == "__main__":
    main()
