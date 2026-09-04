These models were created with project 
git@github.com:nking/recommender_systems.git

the user_movie_model saved_model can be created with
https://github.com/nking/recommender_systems/blob/main/src/test/python/movie_lens_tfx/run_kaggle_pipelines.py

and the metadata_model saved_model can be created with
https://github.com/nking/recommender_systems/blob/main/src/test/python/movie_lens_tfx/run_kaggle_metadata_pipelines.py

=====================================================================

user_movie_model:

signatures (use saved_model_cli to see more):
   - serving_candidate:
     serialized example with "movie_id" and "genres"

   - serving_query:
     serialized example with "user_id", "age", "occupation", "gender", "timestamp"

   - serving_candidate_dict
     expecting named inputs movie_id, genres

   - serving_query_dict:
     expecting named inputs user_id, age, occupation, gender, timestamp

The model is TwoTowerDNN, a bi-encoder trained with 
an in-batch softmax objective (contrastive, listwise loss)
and corrected for item sampling bias 
following Yi et al. 2019 "Sampling-bias-corrected neural modeling
for large corpus item recommendations" and separation of ratings
frequency components.

These metrics are from the pipeline's evaluation on the test
dataset, and uses in-batch negatives.
The metrics performed on the full movie catalog with the
test dataset are performed in this project in the test 
directory.

see metrics and model hyperparameters
  in 
     src/test/resources/serving_model/
  of  
    https://github.com/nking/recommender_systems

=========================================================
The metadata model with batch_size 32 has RMSE 0.25
on the test dataset 

