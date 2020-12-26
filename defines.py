import pandas as pd
from sklearn.neighbors import NearestNeighbors

# dataset
DATASET_PATH = '2017-18_NBA_salary.csv'
SEASON_STATS_DATASET_PATH = 'nba_17_18.csv'
SALARY_PREDICTION_MODEL_PATH = 'salary_model.pkl'
KNN_MODEL_PATH = 'knn_model.pkl'

# base options able to set
DEFAULT_CONTROL_FEATURES = ["NBA_DraftNumber", "Age", "MP", "PER", "USG%", "BPM"]
DEFAULT_PICKED_ADVANCED_FEATURES = ["Age", "MP"]
INTEGER_FEATURES = ["NBA_DraftNumber", "Age"]
SEASON_GAMES_COUNT = 82 # i scale MP written as a season playing time to average per game

# set a name for default unfamous player
ABSTRACT_PLAYER = 'Abstract player' 

# define bounds and initial filling for sliders
DEFAULT_FEATURES_RANGE = pd.DataFrame({
    'min': [  1 ,  18 ,   0. , -50.,   0. , -60.],
    'max': [  62 ,  45 , 48. , 150.,   60. ,  60.],
}, index=DEFAULT_CONTROL_FEATURES)

# define minimum salary for model
MODEL_MIN_VALUE = 46080

__FEATURES_EXPLANATION = [
    "Position of player in NBA draft before his first season in league. ",
    "Player age on February 1 of the given season.",
    "Average minutes played per game during the previous season.",
    "Player Efficiency Rate (PER) - The PER sums up all a player's positive accomplishments, \
    subtracts the negative accomplishments, and returns a per-minute rating of a player's performance.",
    "Usage percentage is an estimate of the percentage of team plays used by a player while he was on the floor.",
    "Box Plus/Minus. A box score estimate of the points per 100 possessions that a player contributed \
         above a league-average player, translated to an average team."
]

FEATURES_DOC_STRING = {k: v for k, v in zip(DEFAULT_CONTROL_FEATURES, __FEATURES_EXPLANATION)}


# Extended class to work with kNN pipeline
class KNN(NearestNeighbors):
    def predict(self, X):
        return super().kneighbors(X,  return_distance=False)