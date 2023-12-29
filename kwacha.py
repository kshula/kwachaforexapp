import streamlit as st
import polars as pl  # Import Polars instead of Pandas
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import plotly.graph_objects as go
from PIL import Image

def read_data(file_path):
    # Use Polars to read the CSV file
    df = pl.read_csv(file_path)
    return df

def create_lag_column(df, lag_days=5):
    # Use Polars shift function
    df = df.with_column('SELLING RATE (5 Days Lag)', df['SELLING RATE'].shift(lag_days))
    df = df.drop_na(subset=['SELLING RATE (5 Days Lag)'])  # Drop NA values
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
    # Use Polars syntax for plotting
    fig = go.Figure()

    # Mid Rate
    fig.add_trace(go.Scatter(x=df['DATE'].to_list(), y=df['MID RATE'].to_list(), mode='lines', name='Mid Rate'))

    return fig


def generate_stats_output(df):
    stats_df = df.describe()
    return stats_df

def predict_future_values(model, lag_values, days):
    # Use Polars to create a DataFrame for predictions
    predictions_df = pl.DataFrame({'SELLING RATE (5 Days Lag)': lag_values})
    predictions_df['Prediction'] = predictions_df['SELLING RATE (5 Days Lag)'].apply(lambda x: model.predict([[x]])[0])

    return predictions_df['Prediction'].to_list()

def plot_predictions(last_n_days, predictions_df, days_to_predict):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=last_n_days['DATE'], y=last_n_days['SELLING RATE'], mode='lines', name='Selling Rate (Actual)'))

    for model_name, predictions in predictions_df.items():
        fig.add_trace(go.Scatter(x=pd.date_range(last_n_days['DATE'].iloc[-1], periods=days_to_predict + 1).shift(1),
                                 y=[last_n_days['SELLING RATE'].iloc[-1]] + predictions.tolist(),
                                 mode='lines', name=f'{model_name} (Predicted)', line=dict(dash='dash')))

    return fig

def calculate_moving_averages(df):
    # Use Polars rolling_mean function
    df = df.with_column('15 Day MA', df['MID RATE'].rolling_mean(window=15))
    df = df.with_column('30 Day MA', df['MID RATE'].rolling_mean(window=30))
    df = df.with_column('60 Day MA', df['MID RATE'].rolling_mean(window=60))
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
    # Use Polars pct_change and rolling_std functions
    df = df.with_column('Mid Rate Returns', df['MID RATE'].pct_change())
    df = df.with_column('Volatility', df['Mid Rate Returns'].rolling_std(window=window))
    df = df.drop_na(subset=['Volatility'])  # Drop NA values
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
        # ...
        df_with_averages = calculate_moving_averages(df.copy())
        fig_averages = visualize_moving_averages(df_with_averages)
        st.plotly_chart(fig_averages)

        volatility_window = st.sidebar.slider("Select Volatility Window (Days)", 1, 30, 20)
        df_with_volatility = calculate_volatility(df.copy(), volatility_window)
        fig_volatility = plot_volatility(df_with_volatility)
        st.plotly_chart(fig_volatility)

        st.write("Go to the 'Stats Output' or 'Predictions' pages for more analysis.")
    elif selected_page == "Stats Output":
        # ...
        stats_df = df.describe().to_pandas()  # Convert to Pandas for compatibility
        st.write(stats_df)

        st.write("Go to the 'Visualization' or 'Predictions' pages for more analysis.")
    elif selected_page == "Predictions":
        # ...
        X = df['SELLING RATE (5 Days Lag)'].to_numpy().reshape(-1, 1)
        y = df['SELLING RATE'].to_numpy().reshape(-1, 1)

        lr_model, rf_model, dt_model, xgb_model = train_models(X, y)

        lag_values = df['SELLING RATE (5 Days Lag)'].tail(days_to_predict).to_numpy()
        lr_predictions = predict_future_values(lr_model, lag_values, days_to_predict)
        rf_predictions = predict_future_values(rf_model, lag_values, days_to_predict)
        dt_predictions = predict_future_values(dt_model, lag_values, days_to_predict)
        xgb_predictions = predict_future_values(xgb_model, lag_values, days_to_predict)

        predictions_df = pl.DataFrame({
            'Linear Regression': lr_predictions,
            'Random Forest': rf_predictions,
            'Decision Tree': dt_predictions,
            'XGBoost': xgb_predictions
        })

        last_90_days = df.tail(90).to_pandas()  # Convert to Pandas for compatibility
        fig = plot_predictions(last_90_days, predictions_df, days_to_predict)
        st.plotly_chart(fig)

        st.write(f"Predicted Selling Rate for the Next {days_to_predict} Days:")
        st.write(predictions_df)
        st.divider()

    st.write("Go to the 'About', 'Home', 'Visualization', 'Stats Output', or 'Predictions' pages for more analysis.")


if __name__ == "__main__":
    main()
