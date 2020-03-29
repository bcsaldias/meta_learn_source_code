import tensorflow as tf
import os
import sys
import random
import numpy as np
import csv
from tensor2tensor.data_generators import text_encoder

from generate_episodes import get_hole_and_sup_episodes, get_hole_episodes
'''
Description: Script which contains all data related routines like generating episodes on the fly, loading data, creating dataset iterators, etc.
'''

#base directory path
base_dir = './'

#base data directory path
base_data_dir = os.path.join(base_dir, 'Preprocessed_Data')

#path of the subword vocab file
subword_vocab_filename = os.path.join(base_data_dir, 'subword_vocab.txt')

######################################################################
maxInt = sys.maxsize

while True:
  # decrease the maxInt value by factor 10
  # as long as the OverflowError occurs.
  try:
      csv.field_size_limit(maxInt)
      break
  except OverflowError:
      maxInt = int(maxInt/10)
#####################################################################

def get_vocab_size():
  encoder = text_encoder.SubwordTextEncoder(subword_vocab_filename)
  vocab_size = encoder.vocab_size
  return vocab_size, encoder

def max_length_sequences(seq):
  return max(len(entry) for entry in seq)

def get_sequence_lengths(seq):
  return [len(entry) for entry in seq]

def get_padded_source_and_target(source, target, window_size): # assume source is the previous context
  seq_len_target = get_sequence_lengths(target)
  max_target_len = max_length_sequences(target)
  padded_target = tf.keras.preprocessing.sequence.pad_sequences(target, maxlen=max_target_len, padding='post')
  padded_source = tf.keras.preprocessing.sequence.pad_sequences(source, maxlen=window_size, padding='pre')
  return padded_source, padded_target, np.array(seq_len_target)

def parse_subtokens(subtoken_string, subtoken_pos):
  subtoken_string = subtoken_string.replace("[","").replace("]","")
  if subtoken_string:
    subtoken_string = [int(ent) for ent in subtoken_string.split(',')]
    subtoken_pos = subtoken_pos.replace("[","").replace("]","")
    subtoken_pos_parts = subtoken_pos.split(",")
    subtoken_pos=[]
    for hole_part in subtoken_pos_parts:
        parts = hole_part.split("%%")
        a = int(parts[0].replace("\'",""))
        b = int(parts[1])
        c = int(parts[2])
        d = parts[3]
        e = float(parts[4])
        f = float(parts[5].replace("\'",""))
        subtoken_pos.append((a,b,c,d,e, f))
    if len(subtoken_pos) > 0:
        return subtoken_string, subtoken_pos
  else:
      return (None, None)

def convert_to_np_array(data, window_size):
  hole_window_raw = [x[1] for x in data]
  hole_target_raw = [x[0] for x in data]
  sup_window_raw = [x[3] for x in data]
  sup_token_raw = [x[2] for x in data]

  max_len_seq = 0
  for i in range(len(sup_token_raw)):
      if sup_token_raw[i]:
          len_seq = max_length_sequences(sup_token_raw[i])
          if len_seq > max_len_seq:
              max_len_seq = len_seq

  sup_window, sup_token, seq_len_sup_token = [], [], []
  hole_window, hole_target, seq_len_hole_target = get_padded_source_and_target(hole_window_raw, hole_target_raw, window_size)

  hole_len = len(hole_window)
  for i in range(hole_len):
      if sup_token_raw[i]:
          u_w, _ , s_u_t = get_padded_source_and_target(sup_window_raw[i], sup_token_raw[i], window_size)
          u_t = tf.keras.preprocessing.sequence.pad_sequences(sup_token_raw[i], maxlen=max_len_seq, padding='post')
          sup_window.append(u_w)
          sup_token.append(u_t)
          seq_len_sup_token.append(s_u_t)

  return hole_window, hole_target, seq_len_hole_target, sup_window, sup_token, seq_len_sup_token

def generator_hole(csv_filename, hole_window_size):
  #Open one row of csv file, generate one hole in the file along with the hole window. Goes sequentially over all the holes in the file. One row of the csv represents one file
  def generate_one_example():
    with open(csv_filename, 'r', encoding='utf-8') as csvfile:
      reader = csv.reader(csvfile)
      for row in reader:
        if ''.join(row).strip(): #added check for empty csv row (Vincent reported that on Windows, there might be extra empty lines added)
          proj_id, dir_id, file_id, subtoken_string, subtoken_pos = row
          subtoken_string, subtoken_pos = parse_subtokens(subtoken_string, subtoken_pos)
          if subtoken_string!= None:
            for hole_id in range(len(subtoken_pos)):
              hole_identity = str(proj_id)+"_" + str(dir_id)+"_" + str(file_id)+"_" + str(hole_id)

              target_hole, hole_window = get_hole_episodes((subtoken_string, subtoken_pos), hole_id, hole_window_size)
              if target_hole:
                hole_window_raw = [hole_window]
                hole_target_raw = [target_hole]
                hole_window, hole_target, seq_len_hole_target = get_padded_source_and_target(hole_window_raw, hole_target_raw, hole_window_size)
                yield (hole_window, hole_target, seq_len_hole_target)

  return generate_one_example

def generator_hole_and_sup_random(csv_filename, hole_window_size, sup_window_size, num_sup_tokens, is_eval, num_of_holes_per_file, sup_def, mode):
  #Open one row of csv file, generate one hole in the file along with the corresponding support set and hole window. Selects a random hole in the file. One row of the csv represents one file
  def generate_one_meta_example_random():
    with open(csv_filename, 'r', encoding='utf-8') as csvfile:
      reader = csv.reader(csvfile)
      for row in reader:
        if ''.join(row).strip(): #added check for empty csv row (Vincent reported that on Windows, there might be extra empty lines added)
          proj_id, dir_id, file_id, subtoken_string, subtoken_pos = row
          subtoken_string, subtoken_pos = parse_subtokens(subtoken_string, subtoken_pos)
          if subtoken_string!= None:
            hole_ids = np.random.randint(0, len(subtoken_pos), num_of_holes_per_file)
            for hole_id in hole_ids:
              hole_identity = str(proj_id)+"_" + str(dir_id)+"_" + str(file_id)+"_" + str(hole_id)
              file_meta_examples = []
              target_hole, hole_window, sup_token, sup_window = \
                      get_hole_and_sup_episodes((subtoken_string, subtoken_pos), hole_id, mode, hole_window_size, sup_window_size, num_sup_tokens, sup_def)
              if target_hole:
                if len(sup_token)> 0:
                  file_meta_examples.append((target_hole, hole_window, sup_token, sup_window))

                  hole_window, hole_target, seq_len_hole_target, sup_window, sup_token, seq_len_sup_token=\
                                                     convert_to_np_array(file_meta_examples, hole_window_size)
                  sup_flag = True

                elif len(sup_token)==0:
                  hole_window_raw = [hole_window]
                  hole_target_raw = [target_hole]
                  hole_window, hole_target, seq_len_hole_target = get_padded_source_and_target(hole_window_raw, hole_target_raw, hole_window_size)
                  sup_flag = False
                yield (hole_window, hole_target, seq_len_hole_target, sup_window, sup_token, seq_len_sup_token, hole_identity, sup_flag)
  return generate_one_meta_example_random

def load_data_sup_and_hole(hole_window_size, sup_window_size, num_sup_tokens, dataset_type, num_of_examples, is_eval, num_of_holes_per_file, sup_def, mode):
  if dataset_type == 'train':
    data_filename = os.path.join(base_data_dir, 'episodes_train.csv')
  elif dataset_type == 'val':
    data_filename = os.path.join(base_data_dir, 'episodes_val.csv')
  elif dataset_type == 'test':
    data_filename = os.path.join(base_data_dir, 'episodes_test.csv')

  output_types = (tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.string, tf.bool)
  dataset = tf.data.Dataset.from_generator(generator_hole_and_sup_random(data_filename, hole_window_size, sup_window_size, num_sup_tokens, is_eval, num_of_holes_per_file, sup_def, mode), output_types)
  dataset = dataset.prefetch(1)
  dataset = dataset.take(num_of_examples)
  return dataset

def load_data_hole(hole_window_size, hole_batch_size, dataset_type, num_of_examples):
  if dataset_type == 'train':
    data_filename = os.path.join(base_data_dir, 'episodes_train.csv')
  elif dataset_type == 'val':
    data_filename = os.path.join(base_data_dir, 'episodes_val.csv')
  elif dataset_type == 'test':
    data_filename = os.path.join(base_data_dir, 'episodes_test.csv')

  output_types = (tf.int32, tf.int32, tf.int32)
  padded_shapes = ([None, None], [None, None], [None])
  dataset = tf.data.Dataset.from_generator(generator_hole(data_filename, hole_window_size), output_types)
  dataset = dataset.shuffle(2000)
  dataset = dataset.padded_batch(hole_batch_size, padded_shapes, drop_remainder = False)
  dataset = dataset.prefetch(1)
  if num_of_examples !='full':
      dataset = dataset.take(num_of_examples)
  return dataset

def getData(hole_window_size, num_examples, dataset_type, sup_window_size=200, num_sup_tokens=512, num_of_holes_per_file=1, sup_def='vocab', mode='tssa', is_eval=True, data_type='hole_and_sup', hole_batch_size=1):
  if data_type == 'hole':
    dataset = load_data_hole(hole_window_size, hole_batch_size, dataset_type, num_examples)
    print("Loaded " + dataset_type +" Hole Data")
  elif data_type == 'hole_and_sup':
    dataset = load_data_sup_and_hole(hole_window_size, sup_window_size, num_sup_tokens, dataset_type, num_examples, is_eval, num_of_holes_per_file, sup_def, mode)
    print("Loaded " + dataset_type+ " Hole and Support Token Data")
  return dataset


