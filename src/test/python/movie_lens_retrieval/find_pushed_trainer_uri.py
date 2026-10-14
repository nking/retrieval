from typing import Tuple

from ml_metadata.proto import metadata_store_pb2
from ml_metadata.metadata_store import metadata_store
import os
import sys
from pathlib import Path

"""
make sure venv has ml-metadata

USAGE example from command line:
export PIPELINE_DATA_DIR
trainer_uri=`python3 find_pushed_trainer_uri.py`
echo $trainer_uri

USAGE example from a python script:
from find_pushed_trainer_uri import get_pushed_trainer_uri
trainer_uri=get_pushed_trainer_uri()
"""

is_cli = __name__ == '__main__' or any('unittest' in arg for arg in sys.argv)
if is_cli:
    sys.path.insert(0,
        os.path.join(os.getcwd(), "src/test/python/movie_lens_retrieval"))
    sys.path.insert(0,
        os.path.join(os.getcwd(), "src/main/python/movie_lens_retrieval"))

from helper import *

def get_pushed_trainer_uri(update_mlmd_pipeline_base:bool=True) -> Tuple[str, str]:
    """
    using the environment variable PIPELINE_DATA_DIR, finds the pushed model in that pipeline's MLMD data
    and finds tehe Trainer model directory origins of the pushed model.
    
    :param update_mlmd_pipeline_base: if True, replaces the pusher_uri and trainer_uri pipeline bases with the
    environ variabel PIPELE_DATA_DIR.  This is needed when the pipele was run on Kaggle for instance and
    downloaded ot a local disk.  By default this parameter is True.
    :return: pusher_uri, trainer_uri
    """
    
    PIPELINE_ROOT = os.environ.get("PIPELINE_DATA_DIR",
                  os.path.join(get_project_dir(), "../recommender_systems/bin/rs_pipeline"))
    
    METADATA_PATH = os.path.join(PIPELINE_ROOT, 'tfx_metadata', 'metadata.db')
    
    if not os.path.exists(METADATA_PATH):
        raise ValueError(f"cannot find {METADATA_PATH}")
    
    metadata_connection_config = metadata_store_pb2.ConnectionConfig()
    metadata_connection_config.sqlite.filename_uri = METADATA_PATH
    store = metadata_store.MetadataStore(metadata_connection_config)
    
    max_time = 0
    trainer_uri = None
    pusher_uri = None
    
    pushed_artifacts = store.get_artifacts_by_type('PushedModel')
    if not pushed_artifacts:
        return None
    else:
        latest_pushed = pushed_artifacts[-1]
        
        pusher_uri = latest_pushed.uri
        
        # Find the Pusher execution that outputted it
        p_events = store.get_events_by_artifact_ids([latest_pushed.id])
        pusher_exec_id = [e.execution_id for e in p_events if e.type == metadata_store_pb2.Event.OUTPUT][0]
    
        # Find all inputs to the Pusher execution
        pusher_inputs = store.get_events_by_execution_ids([pusher_exec_id])
        
        for event in pusher_inputs:
            if event.type == metadata_store_pb2.Event.INPUT:
                art = store.get_artifacts_by_id([event.artifact_id])[0]
                art_type = store.get_artifact_types_by_id([art.type_id])[0].name
                
                if art_type == 'ModelBlessing':
                    blessing_timestamp = art.last_update_time_since_epoch
                    blessing_events = store.get_events_by_artifact_ids([art.id])
                    eval_exec_ids = [e.execution_id for e in blessing_events if e.type == metadata_store_pb2.Event.OUTPUT]
                    
                    if eval_exec_ids:
                        eval_inputs = store.get_events_by_execution_ids([eval_exec_ids[0]])
                        for ev_event in eval_inputs:
                            if ev_event.type == metadata_store_pb2.Event.INPUT:
                                in_art = store.get_artifacts_by_id([ev_event.artifact_id])[0]
                                in_art_type = store.get_artifact_types_by_id([in_art.type_id])[0].name
                                if in_art_type == 'Model':
                                    #print(in_art)
                                    #print("=====")
                                    if in_art.last_update_time_since_epoch > max_time:
                                        trainer_uri = in_art.uri
                                        max_time = in_art.last_update_time_since_epoch
    
    if update_mlmd_pipeline_base:
        #/kaggle/working/recommender_systems/bin/rs_pipeline/Trainer/model/19
        #/kaggle/working/recommender_systems/bin/rs_pipeline
        pipeline_dir = Path(PIPELINE_ROOT).parts[-1]
        idx = pusher_uri.find(pipeline_dir)
        if idx > -1:
            pusher_uri = pusher_uri.replace(pusher_uri[0:idx + len(pipeline_dir)], PIPELINE_ROOT)
        idx = trainer_uri.find(pipeline_dir)
        if idx > -1:
            trainer_uri = trainer_uri.replace(trainer_uri[0:idx + len(pipeline_dir)], PIPELINE_ROOT)
    
    return pusher_uri, trainer_uri

if __name__ == '__main__':
    pusher_uri, trainer_uri = get_pushed_trainer_uri()
    print(pusher_uri, trainer_uri)
