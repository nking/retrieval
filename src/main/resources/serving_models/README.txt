These models were created with project 
git@github.com:nking/recommender_systems.git

the user_movie_model saved_model can be created with
https://github.com/nking/recommender_systems/blob/main/src/test/python/movie_lens_tfx/run_kaggle_pipelines.py

and the metadata_model saved_model can be created with
https://github.com/nking/recommender_systems/blob/main/src/test/python/movie_lens_tfx/run_kaggle_metadata_pipelines.py

=====================================================================

user_movie_model:

use_bias_cor was fixed to True and the Tuner used keras_tuner.Hyberband
to find the best hyper-parameters.

The metrics calculated on test dataset:
   val_hit_rate = 0.00188
   normalized for batch_size=1024 gives NHR = 1.93
   which is better than an NHR of 1 for random.

hyper-parameters:
learning_rate: 0.0021514976439422134
weight_decay: 2.3091699787891865e-05
regl2: 6.7550885719554515e-06
drop_rate: 0.3274279953460161
embed_out_dim: 32
layer_sizes: [32]
feature_acronym: ahosy
incl_genres: True
BATCH_SIZE: 1024
NUM_EPOCHS: 20
use_bias_corr: True
bias_corr_alpha: 0.05
temperature: 0.1
n_users: 6040
n_movies: 3952
n_genres: 18
run_eagerly: False
device: CPU
MAX_TUNE_TRIALS: 10
EXECUTIONS_PER_TRIAL: 1
num_train: 370838
num_eval: 46354
version: 1.0.0
model_name: user_movie
Score: 0.001881777192465961

=========================================================
The metadata model with batch_size 32 has RMSE 0.25
on the test dataset (which a train, val, test split of the train dataset).

