these models were created by scripts
src/test/python/movie_lens_tfx/run_kaggle_pipelines.py
and
src/test/python/movie_lens_tfx/run_kaggle_metadata_pipelines.py
in project https://github.com/nking/recommender_systems.git

Both models are Bi-Encoders with signatures for Query and Candidate
models which can be used to make embeddings that can be used
in approximate nearest neighbo searches.
 * user_movie_model is a Listwise Discriminitive model trained
   using contrastive learning.
 * metadata_model is a regression model trained 
   using mean square error of rating.
