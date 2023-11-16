import pandas as pd
import numpy as np
from ipywidgets import interact,interactive, widgets
import seaborn as sns
import matplotlib.pyplot as plt


all_games = pd.read_csv('df1_games.csv')

game_titles = list(all_games['name'])

all_games.dropna(inplace=True)

all_games['genre/tags'] = all_games[['genre','tags']].agg(''.join,axis=1)

filtered_games = all_games[all_games['positive_rating'] >= 70].reset_index()

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

genre_matrix = cv.fit_transform(filtered_games['genre/tags'])

from sklearn.metrics.pairwise import cosine_similarity

genre_cosine = cosine_similarity(genre_matrix)

def get_games(game):

    idx = filtered_games[filtered_games['name'].apply(lambda x: x.lower()) == game].index[0]
    sim_scores = list(enumerate(genre_cosine[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    get_games = [i[0] for i in sim_scores]



    return filtered_games['name'].iloc[get_games]

def turn_to_percent(game):
    idx = filtered_games[filtered_games['name'].apply(lambda x: x.lower()) == game].index[0]
    sim_scores = list(genre_cosine[idx])
    sim_scores = sorted(sim_scores, reverse=True)
    sim_scores = sim_scores[1:6]

    return sim_scores

