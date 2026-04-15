from setuptools import setup, find_packages

setup(
  name='movie_lens_retrieval',
  version='0.1.0',
  packages=find_packages(where="src/main/python",
    include=['movie_lens_retrieval']),
  package_dir={'': 'src/main/python'},
  install_requires = [
    'faiss-cpu; platform_system != "Linux"',
    'scann; platform_system == "Linux"',
    'tensorflow>=2.20.0', 'numpy >= 2.0',
    'array-record=0.8.3', 'msgpack==1.1.2', 'msgpack-numpy==0.4.8',
    'polars>=1.35.2'
  ],
  extras_require={"test": ["pytest"]},
  classifiers=[ 'Natural Language :: English',
               'Programming Language :: Python :: 3.10',
               'Programming Language :: Python :: 3.11',
               'Programming Language :: Python :: 3.12',
               'Programming Language :: Python :: 3.13',
               'Development Status :: 1 - Development/Unstable'
  ],
  url='https://www.kaggle.com/code/nicholeasuniquename/recommender-systems/',
  license='MIT',
  author='Nichole King',
  author_email='',
  description='Retrieval for Kaggle recommender systems project'
)
