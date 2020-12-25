import plotly.express as px
import pandas as pd


"""
df here - already sorted by Salary with all required fields
"""

def plot_salaries_hist(df):
    fig = px.bar(
        df,
        x='Player', 
        y='Salary',
        template='ggplot2',
        title='Salaries of top5 closest players')
    return fig

def plot_power_angle(df):
    df = df[["Player", "Salary", "PER", "USG%", "BPM"]]

    player_skills = df.drop(['Salary'], axis=1)
    polar_df = pd.melt(player_skills, id_vars=['Player'], 
                   value_vars=player_skills.drop('Player', axis=1).columns, var_name='Skills')
    
    fig = px.line_polar(polar_df, r="value", theta="Skills", color='Player', line_close=True,
                    color_discrete_sequence=px.colors.sequential.Plasma_r,
                    template="gridon", 
                    animation_frame="Player",
                    animation_group="Player",
                    title='Skills distribution for nearest players'
                   )
    return fig