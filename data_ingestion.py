import pandas as pd
import numpy as np
import requests
import io

SEASONS = ['2023-24', '2024-25']
BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"

REQUIRED_COLUMNS = [
    'name', 'kickoff_time', 'element', 'opponent_team', 'was_home',
    'minutes', 'total_points', 'goals_scored', 'assists', 'clean_sheets',
    'goals_conceded', 'saves', 'yellow_cards', 'red_cards', 'bonus',
    'influence', 'creativity', 'threat', 'ict_index', 'value', 'difficulty'  
]

def fetch_season_data(season):
    """
    Fetches the merged gameweek data for a specific season directly from GitHub.
    Handles 'bad lines' where the CSV formatting is broken.
    """
    print(f"Fetching data for season: {season}...")
    url = f"{BASE_URL}/{season}/gws/merged_gw.csv"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        df = pd.read_csv(
            io.StringIO(response.text), 
            on_bad_lines='skip',
            engine='python'
        )
        
        df['season'] = season
        print(f"  -> Successfully loaded {len(df)} rows from {season}.")
        return df

    except Exception as e:
        print(f"  -> Error fetching {season}: {e}")
        return pd.DataFrame()

def clean_data(df):
    """
    Performs initial cleaning: filtering 0 minutes, converting types, and sorting.
    """
    print("\nStarting Data Cleaning...")
    
  
    if 'name' in df.columns:
        df['name'] = df['name'].astype(str)

   
    initial_rows = len(df)
    df = df[df['minutes'] > 0].copy()
    print(f"  -> Dropped {initial_rows - len(df)} rows where minutes = 0 (Bench/DNP).")

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        print(f"  -> Warning: Missing columns in source data: {missing_cols}")
    
    cols_to_keep = [col for col in REQUIRED_COLUMNS if col in df.columns]
    df = df[cols_to_keep + ['season']]


    numeric_cols = ['influence', 'creativity', 'threat', 'ict_index', 'total_points', 'minutes']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

  
    if 'kickoff_time' in df.columns:
        df['kickoff_time'] = pd.to_datetime(df['kickoff_time'])
        # Sort by Player Name first, then by Date (Season -> Gameweek)
        df = df.sort_values(by=['name', 'kickoff_time'])
    
    return df

def main():
    all_data = []
    
    for season in SEASONS:
        season_df = fetch_season_data(season)
        if not season_df.empty:
            all_data.append(season_df)
    
    if not all_data:
        print("No data found! Check your internet connection or the GitHub URL.")
        return

    full_df = pd.concat(all_data, ignore_index=True)
    
    cleaned_df = clean_data(full_df)
    
    output_filename = 'fpl_data_23_25.csv'
    cleaned_df.to_csv(output_filename, index=False)
    print(f"\nSUCCESS: Data saved to {output_filename} ({len(cleaned_df)} rows).")
    print("You are now ready for Feature Engineering.")

if __name__ == "__main__":
    main()