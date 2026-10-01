"""
scripts to analysis the trained models from the recommender_systems project.

some of the analysis is in rust in the ranker project directory inference_src and will be ported
to python here to have it all in one place
"""
import json
import os
import subprocess
import argparse

# 0 = all logs, 1 = no info, 2 = no info/warn, 3 = no info/warn/error
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="umap")


import sys
is_cli = __name__ == '__main__' or any('unittest' in arg for arg in sys.argv)
if is_cli:
    #print('** running from CLI **', flush=True)
    sys.path.insert(0,
        os.path.join(os.getcwd(), "src/test/python/movie_lens_retrieval"))
    sys.path.insert(0,
        os.path.join(os.getcwd(), "src/main/python/movie_lens_retrieval"))

'''
example usages from CLI
 Use either --analyze_and_compare or --compare
 
python3 src/test/python/movie_lens_retrieval/post_training_compare_models.py \
--analyze_and_compare \
--SAVED_MODEL_DIR_1=../TMP10/bin/rs_pipeline/Pusher/pushed_model \
--MODEL_VERSION_1="21" \
--OUTPUT_BASE_DIR_1="../TMP10" \
--SAVED_MODEL_DIR_2=../TMP11/bin/rs_pipeline/Pusher/pushed_model \
--MODEL_VERSION_2="21" \
--OUTPUT_BASE_DIR_2="../TMP11" \
--quiet

python -m debugpy --listen 5678 --wait-for-client \
src/test/python/movie_lens_retrieval/post_training_compare_models.py \
--analyze_and_compare \
--SAVED_MODEL_DIR_1=../TMP10/bin/rs_pipeline/Pusher/pushed_model \
--MODEL_VERSION_1="21" \
--OUTPUT_BASE_DIR_1="../TMP10" \
--SAVED_MODEL_DIR_2=../TMP11/bin/rs_pipeline/Pusher/pushed_model \
--MODEL_VERSION_2="21" \
--OUTPUT_BASE_DIR_2="../TMP11" \
--quiet

or just compare:

python3 src/test/python/movie_lens_retrieval/post_training_compare_models.py \
--compare \
--INPUT_BASE_DIR_1="../TMP10/post_training_analysis" \
--INPUT_BASE_DIR_2="../TMP11/post_training_analysis"
'''

from helper import get_project_dir, get_bin_dir

def run_analysis(saved_model_dir: str, model_version: str, output_dir: str,
        quiet: bool = False) -> str:
    script_path = os.path.join(get_project_dir(),
        "src/test/python/movie_lens_retrieval/post_training_analysis.py")
    # export SAVED_MODEL_DIR="../TMP10/bin/rs_pipeline/Pusher/pushed_model"
    # export MODEL_VERSION="21"
    # export OUTPUT_BASE_DIR="../TMP10"
    # python3 -m unittest src.test.python.movie_lens_retrieval.post_training_analysis.TestAnalysis
    env = os.environ.copy()
    env["SAVED_MODEL_DIR"] = saved_model_dir
    env["MODEL_VERSION"] = model_version
    env["OUTPUT_BASE_DIR"] = output_dir
        
    stdout_setting = subprocess.DEVNULL if args.quiet else None
    
    result = subprocess.run(["python3", script_path], env=env,
        stdout=stdout_setting)
    if result.returncode == 0:
        print("model analyzed")
    else:
        print("model analysis failed.")
    
    return os.path.join(output_dir, "post_training_analysis")
   
def run_compare(dir_path_1:str, dir_path_2:str, quiet:bool=False) -> str:
    
    file_path_1 = os.path.join(dir_path_1, "summary_metrics.json")
    file_path_2 = os.path.join(dir_path_2, "summary_metrics.json")

    with open(file_path_1, "r") as f:
        metrics_1 = json.load(f)
        
    with open(file_path_2, "r") as f:
        metrics_2 = json.load(f)
        
    print(metrics_2)
   
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description="Compare retrieval analysis of 2 models")
    parser.add_argument(
        "--analyze_and_compare",
        action="store_true",
        help="analyze and compare the retrieval from 2 bi-encoders."
             "Note that if set, requires SAVED_MODEL_DIR_1, MODEL_VERSION_1, OUTPUT_BASE_DIR_1 and the same with _2 endings for the 2nd model",
        required=False
    )
    for v in ("1", "2"):
        parser.add_argument(
            f"--SAVED_MODEL_DIR_{v}", type=str,
            help="path to saved_model directory.  path should not contain version.",
            required=False
        )
        parser.add_argument(
            f"--MODEL_VERSION_{v}", type=str,
            help=f"The version for SAVED_MODEL_DIR_{v}",
            required=False
        )
        parser.add_argument(
            f"--OUTPUT_BASE_DIR_{v}", type=str,
            help="the base directory to which a direcotry called post_training_analysis will be written",
            required=False
        )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="compare the retrieval analysis of 2 models"
             "Note that if set, requires INPUT_BASE_DIR_1 and INPUT_BASE_DIR_2 to be set and both must contain a file called 'summary_metrics.json'",
        required=False
    )
    for v in ("1", "2"):
        parser.add_argument(
            f"--INPUT_BASE_DIR_{v}", type=str,
            help=f"path to model {v} directory containing a file called 'summary_metrics.json'",
            required=False
        )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress stdout from the test script"
    )
    
    args = parser.parse_args()
    args_dict = vars(args)
    
    quiet = args_dict.get("quiet")
    
    if args_dict.get("analyze_and_compare"):
        saved_model_dir_1 = args_dict.get("SAVED_MODEL_DIR_1")
        saved_model_dir_2 = args_dict.get("SAVED_MODEL_DIR_2")
        model_version_1 = args_dict.get("MODEL_VERSION_1")
        model_version_2 = args_dict.get("MODEL_VERSION_2")
        output_dir_1 = args_dict.get("OUTPUT_BASE_DIR_1")
        output_dir_2 = args_dict.get("OUTPUT_BASE_DIR_2")
        
        input_base_dir_1 = run_analysis(saved_model_dir_1, model_version_1, output_dir_1, quiet=quiet)
        print(f'wrote to {input_base_dir_1}')
        input_base_dir_2 = run_analysis(saved_model_dir_2, model_version_2, output_dir_2, quiet=quiet)
        print(f'wrote to {input_base_dir_2}')
    
    elif args_dict.get("compare"):
        input_base_dir_1 = args_dict.get("INPUT_BASE_DIR_1")
        input_base_dir_2 = args_dict.get("INPUT_BASE_DIR_2")
    
    else:
        print("choose --analyze_and_compare or --compare", flush=True)
        exit(0)
    
    print("begin comparison")
    output_dir = run_compare(input_base_dir_1, input_base_dir_2, quiet=quiet)
    
