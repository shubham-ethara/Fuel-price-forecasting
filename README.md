# ⛽ Fuel Price Forecasting System

A Python-based Fuel Price Forecasting Application built using Machine Learning and Streamlit.

This system analyzes historical petrol and diesel prices from the National Data & Analytics Platform (NDAP) and forecasts future prices for major Indian metro cities.

---

## 🚀 Project Overview

The Fuel Price Forecasting System:

- Loads official NDAP metro fuel price dataset
- Cleans and preprocesses time-series data
- Performs feature engineering (lag, rolling mean, pct change)
- Trains regression models (Linear Regression & Random Forest)
- Evaluates model performance (MAE, RMSE, R²)
- Forecasts future fuel prices
- Provides an interactive Streamlit dashboard
- Supports city-wise forecasting (Delhi, Mumbai, Chennai, Kolkata)

---

## 📊 Dataset Information

**Dataset Name:** NDAP Retail Selling Price (RSP) of Petrol and Diesel in Metro Cities  
**Source:** National Data & Analytics Platform (NDAP)  
**Format:** CSV  
**File Used:** `NDAP_REPORT_7916.csv`

The dataset includes:

- Calendar Day (Date)
- Product (Petrol / Diesel)
- Metro City
- Retail Selling Price (INR/Litre)

---

## 🧠 Machine Learning Models Used

The system implements:

1. **Linear Regression**
2. **Random Forest Regressor**

Each fuel type (Petrol & Diesel) is trained separately.

### 📈 Evaluation Metrics

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## 🔮 Forecasting Logic

The system performs **recursive multi-step forecasting**:

- Uses latest available data
- Updates lag features dynamically
- Updates rolling averages
- Updates percentage change
- Generates predictions month-by-month

This ensures realistic evolving forecasts instead of static flat predictions.

---

## 🏗️ Project Structure
fuel_price_prediction/
│
├── data/
│ └── NDAP_REPORT_7916.csv
│
├── src/
│ ├── config.py
│ ├── data_loader.py
│ ├── feature_engineering.py
│ ├── model.py
│ └── forecasting.py
│
├── app.py
├── main.py
├── requirements.txt
└── README.md


---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone <your-repo-url>
cd fuel_price_prediction

2️⃣ Create Virtual Environment

python -m venv env
source env/bin/activate   # Linux/Mac

On Windows:
env\Scripts\activate

3️⃣ Install Dependencies

pip install -r requirements.txt

▶️ Running the Application

Run Backend (CLI Mode)
python main.py

Run Streamlit Dashboard
streamlit run app.py

The application will open in your browser.


🖥️ Streamlit Dashboard Features

Metro city selection
Historical fuel price visualization
Forecast horizon selection (1–24 months)
Forecast table output
Forecast chart visualization
Model evaluation metrics display


🛠️ Technologies Used

Python
Pandas
NumPy
Scikit-learn
Streamlit
Matplotlib


🔮 Future Enhancements

TimeSeriesSplit for time-aware validation
Model persistence (joblib)
JSON forecast export
Deployment on Streamlit Cloud
Integration of external economic indicators
ARIMA / Prophet integration