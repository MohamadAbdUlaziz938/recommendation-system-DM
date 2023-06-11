import numpy as np
import pandas as pd
import math
import operator

#Class for Item similarity based Recommender System model
class similarityUser():
    def __init__(self):
        self.train_data = None
        self.recommendations={}
        
    def get_common_book(self,criticA,criticB):
        return [book for book in self.recommendations[criticA] if book in self.recommendations[criticB]]
    
    def create_dictionary(self, train_data):
        self.recommendations={}
        self.train_data=train_data
        for i,user_id in enumerate(train_data["user_id"]):

            for j,book_id in enumerate(train_data["book_id"][i:i+1]):
                bookId=book_id
            for j,Rating in enumerate(train_data["rating"][i:i+1]):
                rating=Rating
            try:
                self.recommendations[user_id][book_id]=rating
            except:
                self.recommendations[user_id]={}      
                self.recommendations[user_id][book_id]=rating
          

        return self.recommendations
    def get_reviews(self,criticA,criticB):
        common_books = self.get_common_book(criticA,criticB)
        return [(self.recommendations[criticA][book], self.recommendations[criticB][book]) for book in common_books]        
    
    # Function to get Euclidean Distance b/w 2 points 
    def euclidean_distance(self,points):
        squared_diffs = [(point[0] - point[1]) ** 2 for point in points]
        summed_squared_diffs = sum(squared_diffs)
        distance = math.sqrt(summed_squared_diffs)
        return distance
    
    # Function to  calculate similarity more similar less the distance and vice versa
    # Added 1 for if highly similar can make the distance zero and give NotDefined Error
    def similarity(self,reviews):
        return 1/ (1 + self.euclidean_distance(reviews))
    
    # Function to get similarity b/w 2 users
    def get_critic_similarity(self,criticA, criticB):
        reviews = self.get_reviews(criticA,criticB)
        return self.similarity(reviews)
    
    
    # Function to give recommendation to users based on their reviews.
    def recommend_books(self,critic, num_suggestions):
        similarity_scores = [(self.get_critic_similarity(critic, other), other) for other in self.recommendations if other != critic]
        # Get similarity Scores for all the critics
        similarity_scores.sort() 
        similarity_scores.reverse()
        similarity_scores = similarity_scores[0:num_suggestions]
        #print(similarity_scores)
        critic_recommendations= {}
        # Dictionary to store recommendations
        for similarity, other in similarity_scores:
            reviewed = self.recommendations[other]
            #print(reviewed)
            # Storing the review
            for book in reviewed:
                if book not in self.recommendations[critic]:
                    #print(book)
                    weight = similarity * reviewed[book]
                    # Weighing similarity with review
                    if book in critic_recommendations:
                        sim, weights = critic_recommendations[book]
                        critic_recommendations[book] = (sim + similarity, weights + [weight])
                        # Similarity of book along with weight
                    else:
                        critic_recommendations[book] = (similarity, [weight])


        for recommendation in critic_recommendations:
            similarity, book = critic_recommendations[recommendation]
            critic_recommendations[recommendation] = sum(book) / similarity
            # Normalizing weights with similarity

        sorted_recommendations = sorted(critic_recommendations.items(), key=operator.itemgetter(1), reverse=True)
        #Sorting recommendations with w
        return similarity_scores,sorted_recommendations



