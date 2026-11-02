import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

INPUT_FILE = 'fpl_data_23_25.csv'
SEQUENCE_LENGTH = 3
FEATURES = [
    'minutes', 'total_points', 'goals_scored', 'assists', 
    'clean_sheets', 'goals_conceded', 'saves', 'bonus', 
    'influence', 'creativity', 'threat', 'ict_index', 'value', 'was_home'
]
TARGET = 'total_points'

def calculate_opponent_difficulty(df):
    """
    Calculates difficulty based on how many points teams concede on average.
    Returns the dataframe with a new 'difficulty' column (0-1 scale).
    """
    print("Calculating implied difficulty from opponent stats...")
    
    df_list = []
    
    for season in df['season'].unique():
        season_df = df[df['season'] == season].copy()
        
        
        opp_stats = season_df.groupby('opponent_team')['total_points'].mean()
        
      
        season_df['avg_points_allowed'] = season_df['opponent_team'].map(opp_stats)
        
    
        season_df['difficulty'] = pd.qcut(season_df['avg_points_allowed'], q=5, labels=[5, 4, 3, 2, 1])
        
        df_list.append(season_df)
    
    return pd.concat(df_list).sort_values(by=['name', 'kickoff_time'])

def create_sequences(df, seq_length):
    sequences = []
    targets = []
    
    player_groups = df.groupby('name')
    
    for player_name, group in player_groups:
        group = group.sort_values('kickoff_time')
        
        data_values = group[FEATURES].values
        target_values = group[TARGET].values
        
        if len(data_values) < seq_length + 1:
            continue
            
        for i in range(len(data_values) - seq_length):
            seq = data_values[i:i+seq_length]
            label = target_values[i+seq_length]
            sequences.append(seq)
            targets.append(label)
            
    return np.array(sequences), np.array(targets)

def main():
    print("Loading data...")
    df = pd.read_csv(INPUT_FILE)
    
    df = calculate_opponent_difficulty(df)
    
    if 'was_home' in df.columns:
        df['was_home'] = df['was_home'].astype(int)
    df['difficulty'] = df['difficulty'].astype(int)

    # 2. SCALING
    print("Scaling features...")
    scaler = MinMaxScaler()
    df[FEATURES] = scaler.fit_transform(df[FEATURES])
    joblib.dump(scaler, 'scaler.pkl')

    # 3. SPLITTING
    print("Splitting data by season...")
    train_df = df[df['season'] == '2023-24']
    test_df = df[df['season'] == '2024-25']
    
    # 4. SEQUENCING
    print(f"Generating sequences (Length: {SEQUENCE_LENGTH})...")
    X_train, y_train = create_sequences(train_df, SEQUENCE_LENGTH)
    X_test, y_test = create_sequences(test_df, SEQUENCE_LENGTH)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")

    np.save('X_train.npy', X_train)
    np.save('y_train.npy', y_train)
    np.save('X_test.npy', X_test)
    np.save('y_test.npy', y_test)
    print("Preprocessing complete!")

if __name__ == "__main__":
    main()