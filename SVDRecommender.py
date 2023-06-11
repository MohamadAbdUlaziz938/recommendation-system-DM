import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

#Class for Item similarity based Recommender System model
class singular_vector_decompistion():
    def __init__(self):
        self.train_data = None
        self.books_unique=pd.DataFrame()
        self.svd=pd.DataFrame()
        self.preds_df=pd.DataFrame()
        self.pivot=None
        self.R=None
        self.user_ratings_mean=None
        self.R_demeaned=None
        self.U=None
        self.sigma=None
        self.Vt=None
        self.all_user_predicted_ratings=None
        
        self.user_id = None
        self.item_id = None
        self.cooccurence_matrix = None
        self.songs_dict = None
        self.rev_songs_dict = None
        self.item_similarity_recommendations = None
        
  
    def get_index_ofUserId_(self,users_test_common):
        t=self.train_data["user_id"].unique()
        t_list=t.tolist()
        #new_list=[]
#         for i in users_test_common:
#             try:
#                 new_list.append(t_list.index(i))
#             except:
#                 new_list.append(1)
        return t_list.index(users_test_common)
    def create(self, train_data):
        self.train_data = train_data

        self.svd["user_id"] = self.train_data["user_id"]
        self.svd["book_id"] = self.train_data["book_id"]
        self.svd["rating"] = self.train_data["rating"]

        # Filter data to only include users who have reviewed at least 50 books 
#         self.svd['number_of_reviews'] = self.svd['user_id'].groupby(self.svd['rating']).transform('count') 
#         self.svd = self.svd[self.svd['number_of_reviews'] >= 50]
#         self.svd = self.svd.drop('number_of_reviews', 1)
        
        
        self.pivot = self.svd.pivot(index = 'user_id', columns ='book_id', values = 'rating').fillna(0)
        self.pivot.head()
        self.R = self.pivot.as_matrix()
        self.user_ratings_mean = np.mean(self.R, axis = 1)
        self.R_demeaned =self.R - self.user_ratings_mean.reshape(-1, 1)
        
        
        self.U, self.sigma, self.Vt = svds(self.R_demeaned, k = 50)
        self.sigma = np.diag(self.sigma)

        self.all_user_predicted_ratings = np.dot(np.dot(self.U, self.sigma), self.Vt) +     self.user_ratings_mean.reshape(-1, 1)
        self.preds_df = pd.DataFrame(self.all_user_predicted_ratings, columns = self.pivot.columns)
        self.get_books_unique_forRecommendation()
        return self.preds_df
        

    
    def get_books_unique_forRecommendation(self):

        self.books_unique["book_id"]=self.train_data["book_id"]
        self.books_unique=self.books_unique.drop_duplicates(subset='book_id', keep="first")
        return self.books_unique
    
    def recommend_books(self,user_id,num_recommendations=5):
        
        if user_id!=0:
            user_row_number = user_id - 1 
        else :
            user_row_number = user_id 
        sorted_user_predictions = self.preds_df.iloc[user_row_number].sort_values(ascending=False)

        
        id_=self.train_data.user_id[user_id:user_id+1]
        for y,z in enumerate(id_):
            id_=z
        user_data = self.train_data[self.train_data.user_id == (id_)]
        user_full=user_data.sort_values(['rating'], ascending=False)
        #print("")
        print("SVD Recommendation")
        print('User {0} has already rated {1} books.'.format(user_id, user_full.shape[0]))
        print('Recommending the highest {0} predicted ratings books not already rated.'.format(num_recommendations))

        # Recommend the highest predicted rating books that the user hasn't seen yet.
        recommendations = (self.books_unique[~self.books_unique['book_id'].isin(user_full['book_id'])].
             merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
                   left_on = 'book_id',
                  right_on = 'book_id').
             rename(columns = {user_row_number: 'Predictions'}).
             sort_values('Predictions', ascending = False).
                           iloc[:num_recommendations, :-1]
                          )

        return user_full, recommendations



