#Function takes a specific genre name and a percentile and returns the upper percentile of movies in that genre
def get_top_genre(genre, meta,  percentile=0.8):

	#create a meta dataset based on each movie genre separately (for a movie with 3 genres you get 3 columns, one for each genre etc)
	s = meta.apply(lambda x: pd.Series(x['genres']), axis=1).stack().reset_index(level=1, drop=True)
	s.name = 'genre'
	print(s)
	genre_meta = meta.drop('genres', axis=1).join(s)

	#create a dataframe based on genre to work on. Calculate mr and mean_rating to calculate ratings
	frame = genre_meta[genre_meta['genre'] == genre]
	num_ratings = frame[frame['vote_count'].notnull()]['vote_count'].astype('int')
	avg_ratings = frame[frame['vote_average'].notnull()]['vote_average'].astype('int')
	mean_rating = avg_ratings.mean()
	mr = num_ratings.quantile(percentile)

	candidates = frame[(frame['vote_count'] >=mr) & (frame['vote_count'].notnull()) & (frame['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity']]
	candidates['vote_count'] = candidates['vote_count'].astype('int')
	candidates['vote_average'] = candidates['vote_average'].astype('int')
	
	#calculate rating for each candidate item separately
	candidates['calc_rating'] = candidates.apply(lambda x: (x['vote_count']/(x['vote_count']+mr) * x['vote_average']) + (mr/(mr+x['vote_count']) * mean_rating), axis=1)
	candidates = candidates.sort_values('calc_rating', ascending = False).head(100)
	return candidates

#Function filters movies that are beyond a specific percentile of the group in terms of vote counts and returns the tops number of those
def get_top_votes(meta,tops,percentile = 0.9):
	#set calculation parameters
	num_ratings = meta[meta['vote_count'].notnull()]['vote_count'].astype('int')
	avg_ratings = meta[meta['vote_average'].notnull()]['vote_average'].astype('int')
	mean_ratings = avg_ratings.mean()
	mr = num_ratings.quantile(percentile) #threshold percentile for number of ratings required for a movie to appear

	#Filter out all movies that have votes < mr (or none), and no average votes (not enough votes)
	candidates = meta[(meta['vote_count'] >= mr) & (meta['vote_count'].notnull()) & (meta['vote_average'].notnull())][['title','year','vote_count','vote_average', 'popularity','genres']]
	candidates['vote_count'] = candidates['vote_count'].astype('int')
	candidates['vote_average'] = candidates['vote_average'].astype('int')
	
	candidates['calc_rating'] = candidates.apply(lambda x: calculate_rating(x,mr,mean_ratings), axis=1)

	#sort by calculated ratings and keep only the top 100
	candidates = candidates.sort_values('calc_rating', ascending=False).head(tops)
	return candidates


#A simple content filtering function takes in the metadata, a title and the number of recommendations and returns the tops similar movies using TF-IDF scores
def get_similar_movies(meta, title, tops):

	#Create extra usable columns with combined info and populate empty cells with NaN
	#Filling process may print some warnings but works properly
	small_meta = meta[meta['id'].isin(links)]
	small_meta['tagline'] = small_meta['tagline'].fillna('')
	small_meta['full_tags'] = small_meta['overview'] + small_meta['tagline']
	small_meta['full_tags'] = small_meta['full_tags'].fillna('')

	#Use TF-IDF to create a vector matrix using the 'full_tags' documents for each item in small_meta
	term_f = TfidfVectorizer(analyzer='word',ngram_range=(1, 2), min_df=0, stop_words='english')
	term_matrix = term_f.fit_transform(small_meta['full_tags'])

	#Use sklearn's linear_kernel to calculate all cosine similarities instantly within the matrix
	cos_sim = linear_kernel(term_matrix,term_matrix)

	small_meta = small_meta.reset_index()
	movie_titles = small_meta['title']
	tindex = pd.Series(small_meta.index, index = small_meta['title'])

	#factor in the title offered for similarity check and calculate the scores of all movies in terms of similarity
	idc = tindex[title]
	scores = list(enumerate(cos_sim[idc]))
	scores = sorted(scores, key= lambda x: x[1], reverse=True)
	scores = scores[1:tops+1]			#scores[0] is the searching movie itself
	movie_idc = [i[0] for i in scores]	#set index pointers to return the tops similar movies from titles list
	return movie_titles.iloc[movie_idc]	



