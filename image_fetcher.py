import numpy as np 
import pandas as pd 
import imdb
import csv
import re

from imdb import IMDb
import imdb.helpers

#Function to create movie posters directory and fill it with posters.
#Each poster is fetched from the IMDB data base and named after its' respective movie id
import urllib
import os

def create_posters_dataset():
	links = pd.read_csv('./ml-25m/links.csv', index_col=0, dtype={'imdbId': str})
	ia = IMDb()
	movd = "movie_posters"
	try:
		os.mkdir(movd)
		print("Directory ", movd, " created.")
	except FileExistsError:
		print("Directory ", movd, " already exists.")

	for i in links.itertuples():
		if(i.Index <= 200048):  #IMDB server is quite unreliable, causing varius SSL crashes
			continue 			#If the fetcher crashes, input the id number of the last downloaded movie +1 to continue from there
		link = i.imdbId
		movie = ia.get_movie(link)
		if(movie == None):
			continue
		else:	
			pic_name ="./movie_posters/" + str(i.Index) + ".jpg"
			movie_url = movie.get('cover')
			print(pic_name)
			try:				#some of the movie poster urls are broken or unretrievable (0.7% of them approximately)
				urllib.request.urlretrieve(movie_url,pic_name)
			except TypeError:
				print(pic_name," could not be retrieved.")

