import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """
    Load NDAP metro fuel dataset and transform into:

    date | petrol_price | diesel_price | state | city

    This version is optimized specifically for the provided NDAP CSV.
    """

    try:
        # Load CSV
        df_raw = pd.read_csv(path)

        # Clean column names
        df_raw.columns = df_raw.columns.str.strip()

        # Rename important columns
        df = df_raw.rename(columns={
            "Products": "product",
            "Products ": "product",
            "Metro cities": "city",
            "Calendar Day": "date",
            "Retail selling price (rsp) of petrol and diesel (UOM:INR/L(IndianRupeesperLitre)) |Scaling Factor:1": "price"
        })

        # Clean product values
        df["product"] = df["product"].str.strip()

        # Convert date properly (already full date string)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # Convert price safely
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

        # Remove invalid rows
        df = df.dropna(subset=["date", "price"])

        # Pivot: Petrol/Diesel rows → columns
        df_pivot = df.pivot_table(
            index=["date", "city"],
            columns="product",
            values="price"
        ).reset_index()

        # Rename pivoted columns
        df_pivot = df_pivot.rename(columns={
            "Petrol": "petrol_price",
            "Diesel": "diesel_price"
        })

        # Map metro cities to states
        city_state_map = {
            "Delhi": "Delhi",
            "Mumbai": "Maharashtra",
            "Chennai": "Tamil Nadu",
            "Kolkata": "West Bengal"
        }

        df_pivot["state"] = df_pivot["city"]

        # Final clean dataset
        df_clean = (
            df_pivot[
                ["date", "petrol_price", "diesel_price", "state", "city"]
            ]
            .dropna()
            .sort_values("date")
        )

        return df_clean

    except Exception as e:
        raise RuntimeError(f"Error processing NDAP dataset: {e}")