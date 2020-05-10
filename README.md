# hybrid-recommender
A hybrid recommender system along with some utility functions concerning movielens dataset processing and enhancement.

The code in this repository was used to compare traditional hybrid recommenders with simple collaborative filtering, content based filtering and deep learning implementations.

The experiments.py includes some early tries in content-based filtering, along with some simple filters.

The utility.py includes metrics used to evaluate the algorithms or rank the data.

The image_fetcher.py includes a small fetching function that takes in a list of movies with their links and fetches their respective posters from the IMDB server. The dataset was not used eventually, although I have plans to include it in enhanced deep content filtering.

The hybrid.py includes the core code for the hybrid recommender system and is fully functional, provided all dependencies are correctly installed. The dataset used is the enhanced movielens 1M dataset from Kaggle, but any movielens dataset with a metadata.csv file can be parsed.

Dependencies include:
-Pandas dataframe library
-Surprise library
-SKlearn library
-scipy
-seaborn
-nltk

For installation instructions for each library, please refer to their respective github repositories.
This work was created as part of a BSc thesis. (5/2020)
