# retrieval
project to retrieve and rank recommendations, built
upon models and data written from the project:
https://github.com/nking/recommender_systems.git

instructions:
  set up a virtual environment using conda or virtualenv
  with a python version that is >= 3.10.0
  
  activate the virtual environment

  to install the dependencies, the easiest way is to
  install this project:
    pip install --editable .
  else you can find the required libraries in pyproject.toml
  or setup.py

  the unit tests show how to run the code.

Local testing:

  pycharm:

    using right click menu, mark the source tree directory:
      src/main/python

    using right click menu, mark the test tree directory:
      src/test/python/movie_lens_retrieval

    then pycharm tests will correctly resolve paths.

  bash or other shell environment:

    python and pytest can be used from the project's base
    directory
  
Misc:
  the cold start, bayesian shrinkage could redone regularly.
  the code is in the reocmmendations project.
  could use for the "m" estimate of 0.75 quantile,
  data sketches like q-digest or t-digest on live data.
