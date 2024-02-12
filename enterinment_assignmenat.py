# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 15:25:18 2023

@author: Admin
"""


'''                      *Problem statement*
The Entertainment Company, which is an online movie watching 
platform, wants to improve its collection of movies and showcase 
those that are highly rated and recommend those movies to its customer
by their movie watching footprint. For this, the company has collected
the data and shared it with you to provide some analytical insights 
and also to come up with a recommendation algorithm so that it can 
automate its process for effective recommendations. The ratings are 
between -9 and +9.



                      *Business Objectives* 
                      
                            *Maximize*

Maximize the visibility or showcase of movies with high ratings 
(ratings closer to +9) to attract users to watch and enjoy highly-rated content.

                          
                            *Minimize*
Minimize the presence or showcasing of movies with low ratings 
(ratings closer to -9) to avoid recommending movies that users are
less likely to enjoy.

                    
                        *Data Dictionary*

Name of feature             Type                          Relevance

ID                     Quantitative(Nominal)          Irrelevant, does not provide useful information
Titles                 Qualitative(categorical)       Relevant
Category               Qualitative(categorical)       Relevant
Review                 Quantitative                   Relevant

'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

entertainment=pd.read_csv("D:/Data Science/6-Datasets/Entertainment.csv.xls")
entertainment

'''                             *EDA*                            '''
entertainment.columns
#['Id', 'Titles', 'Category', 'Reviews'], dtype='object'
entertainment.shape
#51,4
entertainment.size
#204
entertainment.head()
#Top 5 rows
entertainment.tail()
#Display bottom 5 rows
entertainment.count()
'''
Id          51
Titles      51
Category    51
Reviews     51
'''
entertainment.describe()
'''
 Id    Reviews
count    51.000000  51.000000
mean   6351.196078  36.289608
std    2619.679263  49.035042
min    1110.000000  -9.420000
25%    5295.500000  -4.295000
50%    6778.000000   5.920000
75%    8223.500000  99.000000
max    9979.000000  99.000000

For the given dataset std is not near to 1 and
mean is not between 0 and 1

'''
entertainment.isnull()
#False-no null values
#True-null values are present
entertainment.isnull().sum()
'''
Id          0
Titles      0
Category    0
Reviews     0

No null values present for given dataset
'''

'''Duplicates'''
dup=entertainment.duplicated()
dup
#True-duplicated value present
#False-duplivate value are not present
sum(dup)
#There 0 null values

'''                      *Checking outliers*                       '''
sns.boxplot(entertainment['Id'])
#No outliers present
sns.boxenplot((entertainment['Reviews']))
#No outliers present


'''              *Recommendation System*                          '''
import pandas as pd
entertainment=pd.read_csv("D:/Data Science/6-Datasets/Entertainment.csv.xls")
entertainment.shape
#you will get 51*4 matrix
entertainment.columns
entertainment.Titles
#Here we are considering only Titles
from sklearn.feature_extraction.text import TfidfVectorizer
#This is term frequency inverse document
#Each row is treated as document
tfidf=TfidfVectorizer(stop_words='english')
#It is going to create Tfidfvectorizer 

entertainment['Titles'].isnull().sum()
#Now let us create tfidf_matrix
tfidf_matrix=tfidf.fit_transform(entertainment.Titles)
tfidf_matrix.shape
#You will get 51,90
#It has createdsparse matrix,it means
#that we have 90 Titles
#on this particular matrix,
#we want to do iem based recommendation, if a user has
#watched gadar then you can recommend spershah movie
from sklearn.metrics.pairwise import linear_kernel
#This is for measuring similarity
cosine_sim_matrix=linear_kernel(tfidf_matrix,tfidf_matrix)
#each element of tfidf_matrix is compared
#with each element of tfidf_matrix only
#output will be similarity matrixof size 12294X122294 size
#Here in cosine_sim_matrix
#There are no movie names only index are provided
#we will try to map movie name with movie index given
#for that purpose custom functon is written
entertainment_index=pd.Series(entertainment.index,index=entertainment['Titles']).drop_duplicates()
#We are convering entertainment_index into series format, we want index and corressponding
entertainment_id=entertainment_index['Jumanji (1995)']
entertainment_id
def get_entertainment_recommendations_with_reviews(name, topN):
    
    # sort the titles based on reviews
    sorted_entertainment = entertainment.sort_values(by='Reviews', ascending=False)
    #Top N titles with highest reviews
    top_entertainment_recommendations = sorted_entertainment.head(topN+1)
    
    #It is taking by original index while assigning so
    #Changing the index
    top_entertainment_recommendations.reset_index(drop=True, inplace=True)
    
    print(top_entertainment_recommendations)

#enter your entertainment and number of animes to be recommended
get_entertainment_recommendations_with_reviews('Heat (1995)', topN=5)