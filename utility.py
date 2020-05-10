import numpy as np 
import math

#Function to calculate cosine between two vectors. Returns any value in [-1,1]
#Result of cos(K) can be +1 for K=0 degrees (same angle->similar vectors)
#  0 for K=90 degrees (orthogonal angle -> not similar vectors)
# -1 for K=180 degrees (diametrical angle -> opposite vectors) 
def cosine_similarity(vector1, vector2):
	xx = 0
	xy = 0
	yy = 0
	for i in range(len(vector1)):
		x = vector1[i]
		y = vector2[i]
		xx += (x*x)
		xy += (x*y)
		yy += (y*y)
	return (xy/math.sqrt(xx*yy))

#Function to convert every item of a column into an integer
#Returns NaN for non convertible items
def convert_int(xcolumn):
	try:
		return int(xcolumn)
	except:
		return np.nan


#calculate ratings by using weighted vectors
def calculate_rating(candidate, mr, mean_ratings):
	v = candidate['vote_count']
	u = candidate['vote_average']
	return (v/(v+mr)*u) + (mr/(mr+v)*mean_ratings)

#function takes in a list of tags and filters the ones that exist within list s
def filter_tags(taglist, s):
	tag = []
	for i in taglist:
		if i in s:
			tag.append(i)
	return tag

#function takes in two lists of predictions vs actual ratings and returns a mean error using RMSE formula
def eval_RMSE(pred_x, actu_y):
	y_actual = np.array(actu_y)
	y_predicted = np.array(pred_x)
	error =(y_actual-y_predicted)**2
	error_mean=float(np.mean(error))
	error_squared=float(math.sqrt(error_mean))
	return error_squared