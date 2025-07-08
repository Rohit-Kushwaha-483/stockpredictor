import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
import os
import joblib
from django.shortcuts import render
from keras.models import load_model

# Load model once at startup
from django.conf import settings
MODEL_PATH = os.path.join(settings.BASE_DIR, 'core', 'Model', 'lstm_model.keras')
model = load_model(MODEL_PATH)

SCALER_PATH = os.path.join(settings.BASE_DIR, 'core', 'Model', 'scaler.pkl')
scaler = joblib.load(SCALER_PATH)

def index(request):
    context = {}

    if request.method == 'POST':
        stock = request.POST.get('stock', 'RELIANCE.NS')  # fallback default

        # Date range
        end = dt.datetime.today()
        start = end - dt.timedelta(days=365 * 10)

        # Fetch data
        try:
            df = yf.download(stock, start=start, end=end)
            if df.empty:
                context['error_msg'] = f"No data found for '{stock}'. Please check the ticker symbol and try again."
                return render(request, 'index.html', context)
        except Exception as e:
            context['error_msg'] = f"An error occurred while fetching data: {e}"
            return render(request, 'index.html', context)

        # EMAs
        ema20 = df['Close'].ewm(span=20, adjust=False).mean()
        ema50 = df['Close'].ewm(span=50, adjust=False).mean()
        ema100 = df['Close'].ewm(span=100, adjust=False).mean()
        ema200 = df['Close'].ewm(span=200, adjust=False).mean()

        # Train/test split
        data_training = pd.DataFrame(df['Close'][:int(len(df) * 0.70)])
        data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):])

        # Scaling        
        data_training_array = scaler.transform(data_training)


        # Prepare input for model
        past_100 = data_training.tail(100)
        final_df = pd.concat([past_100, data_testing], ignore_index=True)
        input_data = scaler.transform(final_df)

        x_test, y_test = [], []
        for i in range(100, len(input_data)):
            x_test.append(input_data[i - 100:i])
            y_test.append(input_data[i, 0])
        x_test = np.array(x_test)
        y_test = np.array(y_test)

        # Predict and rescale
        y_predicted = model.predict(x_test)
        scale_factor = 1 / scaler.scale_[0]
        y_predicted = y_predicted * scale_factor
        y_test = y_test * scale_factor

        # --- Predict today and next day ---
        # Start with the last 100 days in the dataset
        last_100_days = final_df[-100:].values
        last_100_days_scaled = scaler.transform(last_100_days)

        # Predict today's price
        input_seq = last_100_days_scaled.reshape(1, 100, 1)
        today_prediction = model.predict(input_seq)
        today_prediction_rescaled = today_prediction * scale_factor

        # Predict next day's price based on today's prediction
        last_100_days_scaled = np.append(last_100_days_scaled[1:], today_prediction).reshape(100, 1)
        input_seq = last_100_days_scaled.reshape(1, 100, 1)
        next_day_prediction = model.predict(input_seq)
        next_day_prediction_rescaled = next_day_prediction * scale_factor

        # Save predictions to context
        context['today_prediction'] = today_prediction_rescaled[0][0]
        context['next_day_prediction'] = next_day_prediction_rescaled[0][0]

        # --- Charts ---
        static_dir = 'static'

        def save_plot(fig, filename):
            filepath = os.path.join(settings.BASE_DIR, 'static', filename)
            fig.savefig(filepath)
            plt.close(fig)
            return os.path.join('static', filename)

        # Plot 1: EMA 20 & 50
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(df['Close'], 'y', label='Closing Price')
        ax1.plot(ema20, 'g', label='EMA 20')
        ax1.plot(ema50, 'r', label='EMA 50')
        ax1.set_title("Closing Price vs Time (20 & 50 Days EMA)")
        ax1.legend()
        context['plot_ema_20_50'] = save_plot(fig1, 'ema_20_50.png')

        # Plot 2: EMA 100 & 200
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(df['Close'], 'y', label='Closing Price')
        ax2.plot(ema100, 'g', label='EMA 100')
        ax2.plot(ema200, 'r', label='EMA 200')
        ax2.set_title("Closing Price vs Time (100 & 200 Days EMA)")
        ax2.legend()
        context['plot_ema_100_200'] = save_plot(fig2, 'ema_100_200.png')

        # Plot 3: Prediction vs Actual
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ax3.plot(y_test, 'g', label='Original')
        ax3.plot(y_predicted, 'r', label='Predicted')
        ax3.set_title("Prediction vs Original Trend")
        ax3.legend()
        context['plot_prediction'] = save_plot(fig3, 'stock_prediction.png')

        # Save full dataset
        dataset_path = os.path.join(static_dir, f"{stock}_dataset.csv")
        df.to_csv(dataset_path)
        context['dataset_link'] = dataset_path

    return render(request, 'index.html', context)
