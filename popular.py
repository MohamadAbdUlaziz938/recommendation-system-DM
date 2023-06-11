import numpy as np
import pandas as pd

class popular_py():
    def __init__(self):
        self.train_data = None
        self.filtered=None
        self.m=None
        self.C =None
    def weighted_score(self,x):
        m=self.m
        C=self.C
        v = x['number_of_ratings_book_popularity'] 
        R = x['average_rating_book'] 
        # Calculation based on an IMDB formula 
        return (v/(v+m) * R) + (m/(m+v) * C) 
    
    def create(self, train_data):
        self.train_data = train_data
        self.train_data['number_of_ratings_book_popularity'] = self.train_data['book_id'].groupby(self.train_data['book_id']).transform('count') 
        # Create a column for the average rating a book has received called 'average_rating' 
        self.train_data['average_rating_book'] = self.train_data['rating'].groupby(self.train_data['book_id']).transform('mean') 
        # Calculate the average rating for all books 
        self.C = self.train_data['rating'].mean() 
        # Calculate the minimum number of ratings a book needs to receive in order to be included in the model 
        self.m = self.train_data['number_of_ratings_book_popularity'].quantile(0.90) 
        # Filter the dataset based on value m 
        self.filtered = self.train_data.copy().loc[self.train_data['number_of_ratings_book_popularity'] >= self.m]
        # Create a 'score' column and give each book a weighted score 
        self.filtered['score'] = self.filtered.apply(self.weighted_score, axis=1) 
        self.filtered.sample(5)
        return self.filtered
    
    def recommend_books(self,user_id,num_recommendations=5):
        most_popular = self.filtered.sort_values('score', ascending=False)
        most_popular = most_popular.drop_duplicates(subset='book_id', keep="first")
        print("")
        print("popular recommendation")
        print('most popular for user_id {0}.'.format(user_id))
        return most_popular[['book_id', 'number_of_ratings_book_popularity', 'average_rating_book', 'score']].head(num_recommendations)



