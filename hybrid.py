import numpy as np
import math
import time
from utility import convert_int
from utility import calculate_rating
from utility import filter_tags
from utility import eval_RMSE

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import ast 

from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD, accuracy
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split


#Load data from dataset
#dtype = {'genres':str,'vote_count':int,'vote_average':int,'name':str}
prep_start = time.time()
ratings = pd.read_csv('./movie_dataset/ratings_small.csv')	#'ratings.csv' can be used instead

links = pd.read_csv('./movie_dataset/links_small.csv')		#'links.csv' can be used instead
links = links[links['tmdbId'].notnull()]['tmdbId'].astype('int')

meta = pd.read_csv('./movie_dataset/movies_metadata.csv', low_memory=False)
meta.iloc[0:3].transpose()
#To further understand how meta is structured, please run print(meta.columns) or print(meta.info())
tags = pd.read_csv('./movie_dataset/keywords.csv')
tags['id'] = tags['id'].astype('int')

credits = pd.read_csv('./movie_dataset/credits.csv')
credits['id'] = credits['id'].astype('int')

#transpose meta set by sorting it by ratings and by genre
meta['genres'] = meta['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name']for i in x] if isinstance(x, list) else [])

#add a "year" column extracting only year value from release date
meta['year'] = pd.to_datetime(meta['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x!= np.nan else np.nan)
meta['id'] = meta['id'].apply(convert_int)
#print(meta[meta['id'].isnull()])	 this was used to determine which indexed IDs are NaN and must be removed
meta = meta.drop([19730,29503,35587]) #has to be used manually. Pandas.dropna() will not just remove rows with NaN IDs
meta['id'] = meta['id'].astype('int')

#Create an extra merged dataset to be used
x_meta = meta.merge(credits, on='id')
x_meta = x_meta.merge(tags, on='id')
x_meta = x_meta[x_meta['id'].isin(links)]
#Format x_meta adding the total of cast and crew members
x_meta['cast'] = x_meta['cast'].apply(literal_eval)
x_meta['crew'] = x_meta['crew'].apply(literal_eval)
x_meta['keywords'] = x_meta['keywords'].apply(literal_eval)
x_meta['cast_num'] = x_meta['cast'].apply(lambda x: len(x))
x_meta['crew_num'] = x_meta['crew'].apply(lambda x: len(x))


#small function to find the director of a movie, given the crew
def get_director(xmeta):
	for i in xmeta:
		if i['job'] == 'Director':
			return i['name']
	return np.nan

x_meta['director'] = x_meta['crew'].apply(get_director)
#add only the first 3 existing cast members. Apply only existing keywords
x_meta['cast'] = x_meta['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x,list) else [])
x_meta['cast'] = x_meta['cast'].apply(lambda x: x[:3] if len(x) >=3 else x)
x_meta['keywords'] = x_meta['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x,list) else [])
#x_meta now offers for each movie: 3 cast members, crew list, keywords, director name, title, year, etc.

#merge names and surnames for cast and director so each full name is considered a unique word for increased accuracy
x_meta['cast'] = x_meta['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
x_meta['director'] = x_meta['director'].astype('str').apply(lambda x: str.lower(x.replace(" ","")))

#adding extra weight to director name is as simple as adding him a few times in the director list
x_meta['director'] = x_meta['director'].apply(lambda x: [x,x, x])

#create a dataframe of tags consisting of any kind of keyword within the movie descriptions
s = x_meta.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'tags'
s = s.value_counts()
s = s[s > 1]				#only keep tags that are non-unique and can be used for similarities

stemmer = SnowballStemmer('english') #set a stemmer to be used for the tags

#filter tags using filter and the stemmer algorithm
x_meta['keywords'] = x_meta['keywords'].apply(lambda x: filter_tags(x,s))
x_meta['keywords'] = x_meta['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
#again create single unique words without spaces in between for increased accuracy
x_meta['keywords'] = x_meta['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
	
#create an overall column to include all usable tags per movie
x_meta['overall'] = x_meta['keywords'] + x_meta['cast'] + x_meta['director'] + x_meta['genres']
x_meta['overall'] = x_meta['overall'].apply(lambda x: ' '.join(x))

#x_meta.to_csv('extra_metadata.csv') use this to create a viewable csv file of x_meta (process may take a few minutes)
#create vector matrix out of x_meta tags
tag_vectors = CountVectorizer(analyzer='word',ngram_range=(1, 2), min_df=0, stop_words='english')
tag_matrix = tag_vectors.fit_transform(x_meta['overall'])

#turn vector matrix into an angle-score matrix
cos_sim = cosine_similarity(tag_matrix,tag_matrix)
x_meta = x_meta.reset_index()

#Pretraining for CF algorithm using Surprise library SVD
reader = Reader()
data = Dataset.load_from_df(ratings[['userId','movieId', 'rating']], reader)

#using SVD from the surprise library, filtering is done using only the movie-user ID correlations and no content at all
trainingset, testset = train_test_split(data, test_size = .25)

algorithm = SVD()		#Singular Value Decomposition

#cross_validate(algorithm, data, measures=['RMSE', 'MAE'], cv=5, verbose = True) #evaluates the SVD algorithm using RMSE and MAE metrics
#Train the SVD algorithm
algorithm.fit(trainingset)

#predict = algorithm.test(testset)	#these two lines offer an RMSE overall accuracy of the SVD trained algorithm over the dataset
#accuracy.rmse(predict)

prep_end = time.time()
prep_time = prep_end - prep_start



#Enhanced content filtering function works similarly to get_similar_movies, only instead of summaries/tags it uses cast/director names and other data
def get_similar_extra(x_meta, title, tops):
	#create vector matrix out of x_meta tags
	tag_vectors = CountVectorizer(analyzer='word',ngram_range=(1, 2), min_df=0, stop_words='english')
	tag_matrix = tag_vectors.fit_transform(x_meta['overall'])

	#turn vector matrix into an angle-score matrix
	cos_sim = cosine_similarity(tag_matrix,tag_matrix)
	x_meta = x_meta.reset_index()

	#create a title list again for the results just like in get_similar_movies()
	movie_titles = x_meta['title']
	tindex = pd.Series(x_meta.index, index=x_meta['title'])
	idc = tindex[title]
	scores = list(enumerate(cos_sim[idc]))
	scores = sorted(scores, key= lambda x: x[1], reverse=True)
	scores = scores[1:tops*3] 			#scores[0] is the searching movie itself, we keep the top + top + top movies requested to filter them out again
	movie_idc = [i[0] for i in scores]	#set index pointers to return the tops similar movies from titles list		

	#filter chosen movies by popularity
	recs = x_meta.iloc[movie_idc][['title', 'vote_count', 'vote_average', 'year']]
	num_ratings = recs[recs['vote_count'].notnull()]['vote_count'].astype('int')
	avg_ratings = recs[recs['vote_average'].notnull()]['vote_average'].astype('int')
	mean_rating = avg_ratings.mean()
	mr = num_ratings.quantile(0.6)		#find the 0.40 of top popular movies any percentile from 0.50 to 0.80 is effective 
	
	#process is exactly the same as in function get_top_votes with an adjusted percentile
	candidates = recs[(recs['vote_count'] >= mr) & (recs['vote_count'].notnull()) & (recs['vote_average'].notnull())]
	candidates['vote_count'] = candidates['vote_count'].astype('int')
	candidates['vote_average'] = candidates['vote_average'].astype('int')
	candidates['calc_rating'] = candidates.apply(lambda x: calculate_rating(x,mr,mean_rating), axis=1)
	candidates = candidates.sort_values('calc_rating', ascending=False).head(tops)

	return candidates

#Collaborative filtering algorithm using Surprise library, gives an estimated rating of a user for a movie
#To increase speed, .fit() should be called outside the function and not every time.
def collab_f(ratings, user, moviet):
	#pre-training for SVD etc has already been done.		

	predict = algorithm.predict(user,moviet, r_ui=4, verbose = True)
	return predict

#hybrid function that encompasses previous CB and CF functionalities
def hybrid(x_meta, movie_ids,cos_sim, userId, title, tops=10):

	indices = pd.Series(x_meta.index, index=x_meta['title'])
	movie_index = movie_ids.set_index('id')										#dataset with "id" as index
	midc = indices[title]
	movieid = movie_ids.loc[title]['movieId']
	tmbdid = movie_ids.loc[title]['id']

	scores = list(enumerate(cos_sim[midc]))		#use the cosine similarity matrix to calculate the similarity scores for said title
	scores = sorted(scores, key=lambda x: x[1], reverse=True)
	scores = scores[1:tops*3]
	m_index = [i[0] for i in scores] 	#create an index mask according to scores

	movie_mtx = x_meta.iloc[m_index][['title', 'vote_count','vote_average','release_date','id']]
	movie_mtx['predicted_rating'] = movie_mtx['id'].apply(lambda x: algorithm.predict(userId, movie_index.loc[x]['movieId']).est)
	movie_mtx = movie_mtx.sort_values('predicted_rating', ascending=False)
	return movie_mtx.head(tops)

#Function to evaluate the hybrid above
def hybrid_evaluator(x_meta,cos_sim):
	ratings = pd.read_csv('./movie_dataset/ratings_small.csv')
	ratings = ratings.drop('timestamp',axis=1)

	#ratings = ratings.sort_values(by=['userId','rating'], ascending=False)

	#create a movie_ids dataset that will have movieId-title pairs for faster parsing 
	links = pd.read_csv('./movie_dataset/links_small.csv')[['movieId','tmdbId']]
	links['tmdbId'] = links['tmdbId'].apply(convert_int)
	links.columns=['movieId','id']
	movie_ids = links.merge(x_meta[['title','id']], on='id').set_index('title')	#dataset with "title" as index


	#Keep for each user his first rated movie
	users_byfirst = ratings.groupby('userId').first()
	#For each user, recommend 10 movies that he might like.
	#Will be measuring response time per user and overall prediction accuracy
	response_timer = []
	prediction_accuracy = []
	#iterate through each user's favorite movie and propose 10 movies for each
	for i, row in users_byfirst.iterrows():
		movieId = row['movieId'].astype('int')
		user = int(i)
		user_rating = row['rating'].astype('float')
		try:			#some users' favorites movies didn't have enough votes to be included in the dataset so are invalid. (2% of missed evaluations can be ignored)
			title = str(x_meta.iloc[movieId][['title']][0])
			print('User: ', user, ' Title: ', title)
		except:
			print("User: ", user, ' has an invalid favorite.')
			continue
		start_time = time.time()			#response time measurement
		try:			#same as with the above exception, some movies cause bugs during filtering (1.1%).
			results = hybrid(x_meta,movie_ids,cos_sim,user,title)	#run the algorithm and save the list of 10 top recommendations per user
		except:
			print("Encountered a bug in the dataset.")
			continue
		end_time = time.time()
		response_timer.append(float(end_time-start_time))
		print(results)
		print("Response time: ", response_timer[0])
		
		#measure accuracy using RMSE for movies that we know the user has rated and keeping a 99% percent of prediction if he hasn't
		estimates = results.drop(columns=['title','vote_count','vote_average','release_date'])
		estimates.columns=['movieId','predicted_rating']
		estimates['userId'] = user
		#estimates will have an additional ¨rating¨ column. if the column is NaN, RMSE will factor in given movie´s rating.
		estimates = pd.merge(estimates,ratings[['userId','movieId','rating']], on=['userId','movieId'], how='left')
		prediction_vector = list(estimates['predicted_rating'])
		actual_vector = estimates['rating']
		actual_vector = [user_rating if math.isnan(x) else x for x in actual_vector]	#replace NaN with user's target rating

		#calculate RMSE error for the top 10 recommendations between predicted and actual vectors
		error = eval_RMSE(prediction_vector,actual_vector)
		prediction_accuracy.append(float(5-error))		#will eventually calculate the mean accuracy for all predictions
	
	mean_accuracy = (sum(prediction_accuracy)/len(prediction_accuracy))
	mean_accuracy = (mean_accuracy/5)*100
	mean_time = sum(response_timer)/len(response_timer)
	print("Average response time: ", mean_time)
	print("Average accuracy: ", mean_accuracy,"%")
	print("Pretraining time: ",prep_time)
		



hybrid_evaluator(x_meta,cos_sim)
print("Preprocessing time: ", prep_time, " seconds.")
print()

