import tensorflow as tf
import numpy as np
from typing import Dict, Union, List

def convert_dict_inputs_to_tf_features(
  inputs_dict: Dict[str, Union[tf.Tensor, np.ndarray]]) -> List[
  tf.train.Features]:
  """
  given a dictionary of inputs of tensors or numpy arrays, return list of tf.train.Features
  Note this will fail for inputs_dict in graph mode.
  """
  features = []
  batch_size = np.shape(inputs_dict['movie_id'])[0]
  for i in range(batch_size):
    try:
      feature_map = {}
      for key, values in inputs_dict.items():
        value = values[i]
        if isinstance(value, tf.Tensor):
          value = value.numpy()
        if len(value) == 1:
          value = value[0]
        else:
          raise ValueError(
            f'ERROR: correct for:key={key}, values={values}, value={value}')
        if isinstance(value, bytes) or values.dtype in [str, tf.string]:
          f = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[value]))
        elif values.dtype in [float, tf.float32, tf.float64,
                              tf.float16]:
          f = tf.train.Feature(
            float_list=tf.train.FloatList(value=[float(value)]))
        elif values.dtype in [int, bool, tf.int64, tf.int32, tf.int16]:
          f = tf.train.Feature(
            int64_list=tf.train.Int64List(value=[int(value)]))
        else:
          raise ValueError(
            f"evalue.dtype={value.dtype}, but only float, int, and str classes are handled.")
        feature_map[key] = f
    except Exception as ex:
      raise Exception(f"ERROR: {ex}")
    features.append(tf.train.Features(feature=feature_map))
  return features

def convert_dict_inputs_to_tf_examples(inputs_dict: Dict[
  str, Union[tf.Tensor, np.ndarray]]) -> List[tf.train.Example]:
  """
  given a dictionary of inputs of tensors or numpy arrays, return list of tf.train.Example
  """
  features_list = convert_dict_inputs_to_tf_features(inputs_dict)
  return [tf.train.Example(features=features) for features in
          features_list]

def convert_dict_inputs_to_tfexample_ser(
  inputs_dict: Dict[str, Union[tf.Tensor, np.ndarray]]) -> List[bytes]:
  """
  given a dictionary of inputs of tensors or numpy arrays, return list of serialized tf.train.Example
  """
  tf_examples_list = convert_dict_inputs_to_tf_examples(inputs_dict)
  return [tf_examples.SerializeToString() for tf_examples in
          tf_examples_list]
