import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler


# Loading the data from CSV files
players_df = pd.read_csv('./players_list.csv')
matches_df = pd.read_excel('./matches wins.xlsx')

# Preprocessing data. Removing unwanted columns which might cause the model to throw an error (example, string values)
start_col = players_df.columns.get_loc('Gls')
features_df = players_df.iloc[:, start_col:]
features_df = features_df.drop(features_df.columns[0], axis=1)
features_df = features_df.drop(columns=['Minutes_played'], errors='ignore')
features_df['Player'] = players_df['Player']
features_df['Team'] = players_df['Team']
features_df.set_index('Player', inplace=True)


X = []
y = []


# Here we create the data set using team and players data, and prepare the input for the model (difference between two team's strenghts)
for idx, row in matches_df.iterrows():
    team1 = row['Team 1']
    team2 = row['Team 2']
    winner = row['Winner']

    team1_players = features_df[features_df['Team'] == team1].drop(columns=['Team', 'Club', 'Club_country','Age','Pos','Minutes_played','Starts','Match_played', 'Player'], errors='ignore')
    team2_players = features_df[features_df['Team'] == team2].drop(columns=['Team', 'Club', 'Club_country','Age','Pos','Minutes_played','Starts','Match_played', 'Player'], errors='ignore')

    if team1_players.empty or team2_players.empty:
        continue

    # Finding a team's strenght
    team1_sum = team1_players.sum()
    team2_sum = team2_players.sum()

    # Data for input for the model
    diff = (team1_sum - team2_sum).values

    if pd.isna(winner):
        continue  # We don't want to include draw games to avoid inconsistency
    elif winner == team1:
        X.append(diff)
        y.append(1)
    elif winner == team2:
        X.append(diff)
        y.append(0)
    else:
        continue


scaler = StandardScaler()
X = np.array(X)
y = np.array(y)
X_scaled = scaler.fit_transform(X)

# Training part
model = LogisticRegression(max_iter=5000)
model.fit(X_scaled, y)

# Printing learned weights
print("Learned feature weights:")
for feature_name, coef in zip(team1_sum.index, model.coef_[0]):
    print(f"{feature_name}: {coef:.4f}")








# Prediciting the finals match

team1 = "Croatia"
team2 = "Morocco"
team1_players = features_df[features_df['Team'] == team1].drop(columns=['Team', 'Club', 'Club_country','Age','Pos','Minutes_played','Starts','Match_played', 'Player'], errors='ignore')
team2_players = features_df[features_df['Team'] == team2].drop(columns=['Team', 'Club', 'Club_country','Age','Pos','Minutes_played','Starts','Match_played', 'Player'], errors='ignore')

team1_sum = team1_players.sum()
team2_sum = team2_players.sum()
diff = (team1_sum - team2_sum).values.reshape(1, -1)

diff_scaled = scaler.transform(diff)

prob = model.predict_proba(diff_scaled)[0][1]

print(f"\nProbability that {team1} wins against {team2}: {prob:.2%}")

