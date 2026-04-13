The files in data are derived from 
[GroupLens](https://files.grouplens.org/datasets/movielens/)
1-m dataset

They are written in
project: https://github.com/nking/recommender_systems.git
in script src/test/python/movie_lens_tfx/WriteRetrievalInputs.py

Some of the files are larger than 50MB, so they're not committed
to github except as DVC tracking commits.

Some details:
----------
movie_emb_input and user_emb_input directories holds
serialized tf examples in tfrecords with format each
row being a dictionary of 'user_id or 'movie_id' and 
'embedding' as a float array of length 16.

movie_ratings_pivot_table directory
holds tfrecords of a movie ratings pivot table made
from the train and val datasets, counting the ratings
then appended to that is the list of all remaining movie
ids from movies.dat but given ratings of 0 counts for each
rating.
columns are "movie_id", "1", "2", "3", "4", "5"
