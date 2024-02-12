# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 09:02:59 2023

@author: Admin
"""

import pandas as pd
anime=pd.read_csv("D:/Data Science/6-Datasets/anime.csv.xls")
anime.shape
#You will get (12294, 7) matrix
anime.columns
anime.genre
#Here we are considering only genere
from sklearn.feature_extraction.text import TfidfVectorizer
#This is term frequency inverse document
#Each row is treated as document
tfidf=TfidfVectorizer(stop_words='english')
#It is going to create Tfidfvectorizer 


anime['genre'].isnull().sum()
#Thre are 62 null values
#Support one movie has got genre Drama,Romance,...
#There may be many empty spaces
#so let us impute these empty spaces,general is like simpler imputer
anime['genre']=anime['genre'].fillna('general')
anime['genre'].isnull().sum()

#Now let us create tfidf_matrix

tfidf_matrix=tfidf.fit_transform(anime.genre)
tfidf_matrix.shape
#You will get 12294,47
#It has createdsparse matrix,it means
#that we have 47 genre
#on this particular matrix,
#we want to do iem based recommendation, if a user has
#watched gadar then you can recommend spershah movie
from sklearn.metrics.pairwise import linear_kernel
#This is for measuring similarity
cosine_sim_matrix=linear_kernel(tfidf_matrix,tfidf_matrix)
#each element of tfidf_matrix is compared
#with each element of tfidf_matrix is compared
#output will be similartiy matrix of size 12294X12294 size
#Here in cosin_sim_matrix,
#there are no movie names only index are provided
#we will try to map movie name with movie index given
#for that purpose custom functon is written
anime_index=pd.Series(anime.index,index=anime['name']).drop_duplicates()
#We are convering anme_index into series format, we want index and corressponding
anime_id=anime_index['Assassins (1995)']
anime_id
def get_recommendations(Name,topN):
    #topN=10
    #Name='Assassins (1995)'
    anime_id=anime_index[Name]
    
    #we want to capture whole row of given movie
    #name,its score and column id 
    #for that purpose we are applying cosine_sim_matrix to enumerate function..
    #Enumerate function create a object,
    #which we need to create in list form
    #we are using enumerate function,
    #what enumerate does,suppose we have given
    #(2,10,15,18), if we apply to enumerate then it will create a list
    #(0,2,  1,10,  3,15,  4,18)
    cosine_scores=list(enumerate(cosine_sim_matrix[anime_id]))
    #the cosine scores captured, we want to arrange in descending order
    #to that
    #we can recommend top 10 based on highest similarity i.e.score
    #x[0]=index and x[1] is cosine score 
    #we want arrange tupples according to decreasing order
    #of the score not index
    #supporting the cosine_similarity scores based on scores i.e.x[1]
    cosine_scores=sorted(cosine_scores,key=lambda x:x[1], reverse= True)
    #get the scores of top N most similar movies
    #to capture TopN movies,you need to give topN+1
    cosine_scores_N=cosine_scores[0: topN+1]
    #getting the movie index
    anime_idx=[i[0] for i in cosine_scores_N]
    #getting cosine score
    anime_scores=[i[1] for i in cosine_scores_N]
    #we are going to use this information to crate a dataframe
    #create a empty dataframe
    anime_similar_show=pd.DataFrame(columns=['name','score'])
    #assign anime_idx to name column
    anime_similar_show['name']=anime.loc[anime_idx,'name']
    #assign score to score column
    anime_similar_show['score']=anime_scores
    #while assigning values,it is by default capturing original index of ..
    #we want to reset the index
    anime_similar_show.reset_index(inplace=True)
    print(anime_similar_show)
#enter your anime and number of animes to be recommended
get_recommendations('Bad Boys (1995)',topN=10)