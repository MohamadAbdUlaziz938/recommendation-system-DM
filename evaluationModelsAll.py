#Class to calculate precision and recall

import random

class precision_recall_calculator():
    
    def __init__(self, test_data, train_data, pm, is_model,model_type_1,model_type_2):
        self.test_data = test_data
        self.train_data = train_data
        self.user_test_sample = None
        self.user_test_sample_ = None
        self.model_type_1=model_type_1
        self.model_type_2=model_type_2
        
        self.model1 = pm
        self.model2 = is_model
        
        self.ism_training_dict = dict()
        self.pm_training_dict = dict()
        self.test_dict = dict()
    
    #Method to return random percentage of values from a list
    def remove_percentage(self, list_a, percentage):
        k = int(len(list_a) * percentage)
        random.seed(0)
        indicies = random.sample(range(len(list_a)), k)
        new_list = [list_a[i] for i in indicies]
        return new_list
    
    def get_index_ofUserId(self,users_test_common):
        t=self.train_data["user_id"].unique()
        t_list=t.tolist()
        new_list=[]
        for i in users_test_common:
            try:
                print(t_list.index(i)+1)
                
                new_list.append(t_list.index(i)+1)
            except:
                new_list.append(1)
        return new_list
    def get_index_ofUserId_(self,users_test_common):
        t=self.train_data["user_id"].unique()
        t_list=t.tolist()
        new_list=[]
        for i in users_test_common:
            try:
                new_list.append(t_list.index(i))
            except:
                new_list.append(1)
        return new_list
            
    #Create a test sample of users for use in calculating precision
    #and recall
    def create_user_test_sample(self, percentage):
        #Find users common between training and test set
        users_test_and_training = list(set(self.test_data['user_id'].unique()).intersection(set(self.train_data['user_id'].unique())))
        print("Length of user_test_and_training:%d" % len(users_test_and_training))

        #Take only random user_sample of users for evaluations
        self.users_test_sample = self.remove_percentage(users_test_and_training, percentage)
        #print(self.users_test_sample)
        print("Length of user sample:%d" % len(self.users_test_sample))
        self.users_test_sample_=self.get_index_ofUserId_(self.users_test_sample)
        
    #Method to generate recommendations for users in the user test sample
    def get_test_sample_recommendations(self):
        #For these test_sample users, get top 10 recommendations from training set
        self.ism_training_dict = {}
        self.pm_training_dict = {}
        self.test_dict = {}
        user_sim_items_=[]
        
        for i in range(len(self.users_test_sample)):
            user_sim_items=[]
            #Get items for user_id from svd model
            print("")
            print("Getting recommendations for user:%s" % self.users_test_sample[i])
            if self.model_type_2=="similarity" and self.model_type_1=="popularity":
                print("user similarity recommendation")
                print("most recommendation books for user_id:%s" % self.users_test_sample[i])
                users_similarity,user_sim_items_ = self.model2.recommend_books(self.users_test_sample[i],10)

                temp=user_sim_items_[:10]
                for j in range(10):
                    user_sim_items.append(temp[j][0])
                self.ism_training_dict[self.users_test_sample[i]] = user_sim_items
                
                user_sim_items = self.model1.recommend_books(self.users_test_sample[i],10)
                self.pm_training_dict[self.users_test_sample[i]] = list(user_sim_items["book_id"])
            elif self.model_type_2=="svd" and self.model_type_1=="popularity":
                #print("svd recommendation")
                already_Rated,user_sim_items = self.model2.recommend_books(self.users_test_sample_[i],10)
                self.ism_training_dict[self.users_test_sample[i]] = list(user_sim_items["book_id"])
                
                user_sim_items = self.model1.recommend_books(self.users_test_sample[i],10)
                self.pm_training_dict[self.users_test_sample[i]] = list(user_sim_items["book_id"])
            elif self.model_type_2=="svd" and self.model_type_1=="similarity":
                
                print("user similarity recommendation")
                print("most recommendation books for user_id:%s" % self.users_test_sample[i])
                users_similarity,user_sim_items_ = self.model1.recommend_books(self.users_test_sample[i],10)
                temp=user_sim_items_[:10]
                for j in range(10):
                    user_sim_items.append(temp[j][0])
                    
                self.pm_training_dict[self.users_test_sample[i]] = user_sim_items
                
                
                already_Rated,user_sim_items = self.model2.recommend_books(self.users_test_sample_[i],10)
                self.ism_training_dict[self.users_test_sample[i]] = list(user_sim_items["book_id"])
           

            # print(self.ism_training_dict)    
            #print(self.pm_training_dict)
    
            #Get items for user_id from test_data
            test_data_user = self.test_data[self.test_data['user_id'] == self.users_test_sample[i]]
            self.test_dict[self.users_test_sample[i]] = set(test_data_user['book_id'].unique() )
    
    #Method to calculate the precision and recall measures
    def calculate_precision_recall(self):
        #Create cutoff list for precision and recall calculation
        cutoff_list = list(range(1,11))


        #For each distinct cutoff:
        #    1. For each distinct user, calculate precision and recall.
        #    2. Calculate average precision and recall.

        ism_avg_precision_list = []
        ism_avg_recall_list = []
        pm_avg_precision_list = []
        pm_avg_recall_list = []


        num_users_sample = len(self.users_test_sample)
        for N in cutoff_list:
            ism_sum_precision = 0
            ism_sum_recall = 0
            pm_sum_precision = 0
            pm_sum_recall = 0
            ism_avg_precision = 0
            ism_avg_recall = 0
            pm_avg_precision = 0
            pm_avg_recall = 0

            for user_id in self.users_test_sample:
                ism_hitset = self.test_dict[user_id].intersection(set(self.ism_training_dict[user_id][0:N]))
                pm_hitset = self.test_dict[user_id].intersection(set(self.pm_training_dict[user_id][0:N]))
                testset = self.test_dict[user_id]
        
                pm_sum_precision += float(len(pm_hitset))/float(N)
                pm_sum_recall += float(len(pm_hitset))/float(len(testset))

                ism_sum_recall+= float(len(ism_hitset))/float(len(testset))
                ism_sum_precision += float(len(ism_hitset))/float(N)
        
            pm_avg_precision = pm_sum_precision/float(num_users_sample)
            pm_avg_recall = pm_sum_recall/float(num_users_sample)
    
            ism_avg_precision = ism_sum_precision/float(num_users_sample)
            ism_avg_recall = ism_sum_recall/float(num_users_sample)

            ism_avg_precision_list.append(ism_avg_precision)
            ism_avg_recall_list.append(ism_avg_recall)
    
            pm_avg_precision_list.append(pm_avg_precision)
            pm_avg_recall_list.append(pm_avg_recall)
            
        return (pm_avg_precision_list, pm_avg_recall_list, ism_avg_precision_list, ism_avg_recall_list)
     

    #A wrapper method to calculate all the evaluation measures
    def calculate_measures(self, percentage):
        #Create a test sample of users
        self.create_user_test_sample(percentage)
        
        #Generate recommendations for the test sample users
        self.get_test_sample_recommendations()
        
        #Calculate precision and recall at different cutoff values
        #for popularity mode (pm) as well as item similarity model (ism)
        
        return self.calculate_precision_recall()
        #return (pm_avg_precision_list, pm_avg_recall_list, ism_avg_precision_list, ism_avg_recall_list)    