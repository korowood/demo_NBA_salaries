import streamlit as st

import pandas as pd
import numpy as np
import joblib

from defines import *
from plotting import plot_salaries_hist, plot_power_angle, \
                    foo_point_leader, top_20_point

st.set_page_config(layout="wide")

# ignore repeated reading for given dataset
@st.cache
def read_dataset():
    df = pd.read_csv(DATASET_PATH)
    return df[DEFAULT_CONTROL_FEATURES + ['Player', 'Salary']]

@st.cache
def read_stats_dataset():
    df = pd.read_csv(SEASON_STATS_DATASET_PATH)
    return df

@st.cache
def __get_unique_players_list():
    df = read_dataset()
    return df.Player.tolist()

@st.cache
def load_pred_model():
    model = joblib.load(SALARY_PREDICTION_MODEL_PATH)
    return model

@st.cache(allow_output_mutation=True)
def load_knn_model():
    model = joblib.load(KNN_MODEL_PATH)
    return model

def __create_features_initial_filling(player):
    df = read_dataset()
    # more efficient with set instead of list, but this returns similar guys too
    features = {
        feature: DEFAULT_FEATURES_RANGE.loc[feature]['min'] for feature in DEFAULT_CONTROL_FEATURES
        }
    if player != ABSTRACT_PLAYER:
        idx = df[df.Player == player].index[0]
        for feature in DEFAULT_CONTROL_FEATURES:
            features[feature] = df.loc[idx][feature]
        features["MP"] = features["MP"] / SEASON_GAMES_COUNT
    return features
        
def setup_sidebar(player):
    bar = st.sidebar
    is_advanced_mode_checkbox = bar.checkbox('Show advanced settings')
    bar.write(
        """With using this option, you will be able to estimate
        how different player's parameters affect fees cost.
        """)
    features = __create_features_initial_filling(player)
    control_features = DEFAULT_PICKED_ADVANCED_FEATURES
    if is_advanced_mode_checkbox:
        control_features = bar.multiselect( 'Add another feature to control',
           DEFAULT_CONTROL_FEATURES, DEFAULT_PICKED_ADVANCED_FEATURES)
        for feature in control_features:
            if feature in INTEGER_FEATURES:
                features[feature] = bar.slider(
                    feature,
                    int(DEFAULT_FEATURES_RANGE.loc[feature]['min']),
                    int(DEFAULT_FEATURES_RANGE.loc[feature]['max']),
                    int(features[feature]))
            else:
                features[feature] = bar.slider(
                    feature,
                    DEFAULT_FEATURES_RANGE.loc[feature]['min'],
                    DEFAULT_FEATURES_RANGE.loc[feature]['max'],
                    float(features[feature]))
            bar.write(FEATURES_DOC_STRING[feature])
    else:
        pass
    features['MP'] *= SEASON_GAMES_COUNT # reverse value for model
    return features

def setup_page_preview():
    st.title("NBA salaries prediction demo")
    """This is a test model aims to predict NBA player salaries based on their 
    statistic in 2017-2018 season. """
    """Follow the instructions below to calculate a prediction.
    """
    st.title("*1. Enter player name.*")
    """You may choose any active player, or if you want, you could set up a player with your own parameters.
    In this case, you should either select 'Abstract player' or set up advanced settings for any player."""
    #player = st.text_input("", '')
    player = st.selectbox("", [ABSTRACT_PLAYER] + __get_unique_players_list())
    """Optional. You may set player's parameters in the sidebar in the left part of the screen.
    It may help you to understand the importance of different stats for player's salary. """
    st.title("*2. When you're ready, press the button. *")
    return player

def get_salary_prediction(feature_values):
    # some shit stuff, pre feature engineering
    feature_values[0][1] = np.abs(30 - feature_values[0][1])
    # then as it should be
    model = load_pred_model()
    ans = np.expm1(model.predict(feature_values))[0][0]
    ans = max(MODEL_MIN_VALUE, ans)
    return np.round(ans, 2)

def get_nearest_players(player, feature_values):
    model = load_knn_model()
    # get nearest ids
    nearest_players_ids  = model.predict(feature_values)[0]
    # get them 
    neighs = read_dataset().iloc[nearest_players_ids]
    # exclude himself (if needs)
    neighs = neighs[neighs.Player != player].head(5)
    neighs.sort_values(by='Salary', ascending=False, inplace=True)
    return neighs

def create_output(player, feature_values):
    # logically separate field for output
    # perform result based on button click
    if st.button('Predict'):
        # get closest players
        neighs = get_nearest_players(player, feature_values)
        # plot what we decided to plot
        left_col, right_col = st.beta_columns(2)
        with left_col:
            fig = plot_salaries_hist(neighs)
            st.write(fig)
        with right_col:
            fig = plot_power_angle(neighs)
            st.write(fig)
            st.write("*PER - Player efficiency rate*")
            st.write("*BPM - box plus minus*")
            st.write("*USG% - usage percentage*")
            st.write("For details - visit https://www.basketball-reference.com/about/glossary.html")
            # base salary prediction
        predicted_salary = get_salary_prediction(feature_values)
        st.success("Predicted salary for " + player + ": " + str(predicted_salary) + " $")
    st.title("*3. Enjoy! *")

def show_league_stat():
    stats_dataset = read_stats_dataset()
    is_show_league_stat = st.checkbox('Show me some league facts')
    if is_show_league_stat:
        left_col, right_col = st.beta_columns(2)
        with left_col:
            fig = foo_point_leader(stats_dataset)
            st.write(fig)
        with right_col:
            fig = top_20_point(stats_dataset)
            st.write(fig)


def main():
    # setup pages and returns its fields
    player = setup_page_preview()
    num_features_dict = setup_sidebar(player)
    # create train features vector
    feature_values = [[v for _, v in num_features_dict.items()]]
    # create output
    create_output(player, feature_values)
    # display league stat
    show_league_stat()


if __name__ == "__main__":
    main()