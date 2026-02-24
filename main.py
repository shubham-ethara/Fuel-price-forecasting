from src.config import *
from src.data_loader import load_data
from src.feature_engineering import create_features
from src.model import train_models

def main():
    df_raw = load_data(DATA_PATH)
    df_features = create_features(df_raw)

    print("Data loaded and features created successfully.")
    print(df_features.head())


if __name__ == "__main__":
    main()