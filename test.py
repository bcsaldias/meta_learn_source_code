import os
import sys
import time
import argparse
import pickle
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict

#Importing from other modules
import losses
from model import Seq2SeqModel
from data import getData, get_vocab_size

#change this number for running on a different GPU in multi-GPU setting
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

'''
Description: Script which performs evaluation given a pretrained model and dataset
'''

#base directory path
base_dir = './'

#base data directory path
base_data_dir = os.path.join(base_dir, 'Preprocessed_Data')

#the file which stores meta-info about each run of the code
master_meta_info_file = os.path.join(base_dir, 'runs.txt')

TOTAL_TRAIN_FILES = 12934
TOTAL_TEST_FILES = 8268
TOTAL_VAL_FILES = 7185

CONFIDENCE_INTERVAL = 0.95

#seed for easy reproduction of results
np.random.seed(42)

def setup_args():
  """
  Description: Takes in the command-line arguments from user
  """
  parser = argparse.ArgumentParser()

  parser.add_argument("--comment", type=str, default='tssa-fomaml-eval', help="comment used to identify the run")
  parser.add_argument("--out_dir", type=str, default='Outputs/', help="Output Directory")
  parser.add_argument("--model_load_dir", type=str, default='Trained_Models/base_model/', help="Directory from which the model is loaded")
  parser.add_argument("--checkpoint_dir", type=str, default='Models/', help="Directory for checkpoint")
  parser.add_argument("--dataset_type", type=str, default='test', help="the type of dataset: train, test, val")
  parser.add_argument("--load_model", type=bool, default=True, help="Whether to load a pretrained model")
  parser.add_argument("--hole_window_size", type=int, default=200, help="Size of the Context window around the hole targets")
  parser.add_argument("--sup_window_size", type=int, default=200, help="Size of the Context window around the support token")
  parser.add_argument("--num_files", type=str, default=8268, help="Number of files to be taken from dataset. For running over the full dataset, use 8268 for test and 7185 for val")
  parser.add_argument("--sup_batch_size", type=int, default=20, help="Batch size of the support set")
  parser.add_argument("--num_sup_tokens", type=int, default=512, help="Number of support tokens")
  parser.add_argument("--inner_learning_rate", type=float, default=5e-4, help="Learning rate for inner Adam Optimizer")
  parser.add_argument("--method", type=str, default='tssa', help="method of evaluation: tssa, dyn_eval, base_model")
  parser.add_argument("--num_of_updates", type=int, default=16, help="Number of inner updates done per hole target (=k) for TSSA")
  parser.add_argument("--num_of_holes_per_file", type=int, default=5, help="Number of hole targets to be sampled per file")
  parser.add_argument("--sup_def", type=str, default='vocab', help="Definition of support token to be used: vocab, proj, random, unique")

  return parser.parse_args()


def evaluate(model, dataset, method, bar, inner_learning_rate, sup_batch_size, num_of_updates):
  """
  Description: Calculates the average token hole target cross-entropy across the dataset after performing inner updates using support set for each hole target. Both the optimizer state and model parameters
               are reset before calculating the next hole target cross-entropy.
  """
  if CONFIDENCE_INTERVAL == 0.95:
      Z = 1.96
  elif CONFIDENCE_INTERVAL == 0.99:
      Z = 2.58
  token_losses = []
  # dict with key = hole features, value = hole loss
  hole_features = {}

  total_subword_loss = 0.0
  total_token_loss = 0.0
  total_batches = 0

  # storing initial weights of the base model so that they can be restored later
  trained_model_trainable_variables = []
  for entry in model.get_weights():
    trained_model_trainable_variables.append(entry)

  for (batch, (hole_window, hole_target, seq_len_hole_target, sup_window, sup_token, seq_len_sup_token, hole_identity, sup_flag)) in enumerate(dataset):

    #Reset before evaluating each hole target
    y = tf.reshape(tf.Variable(1, dtype=tf.int32), (1,1))
    model(y, y, False)
    model.set_weights(trained_model_trainable_variables)

    if sup_flag:
      sup_window = tf.squeeze(sup_window, axis=0)
      sup_token = tf.squeeze(sup_token, axis=0)
      seq_len_sup_token = tf.squeeze(seq_len_sup_token, axis=0)

    if sup_flag and (method =='tssa' or method == 'dyn_eval'):
      # Get the new model object after performing num_of_updates inner updates
      model_new = losses.inner_loss_eval(model, sup_window, sup_token, seq_len_sup_token, False, method, inner_learning_rate, sup_batch_size, num_of_updates)
      # Calculate the hole target loss using the new model object
      batch_token_loss, masked_loss = losses.hole_loss(model_new, hole_window, hole_target, seq_len_hole_target, False)

    # If there are no support tokens found in the file or if the evaluation mode is for a base_model, directly calculate the hole target loss using the hole window
    if not sup_flag or method=='base_model':
      batch_token_loss, masked_loss = losses.hole_loss(model, hole_window, hole_target, seq_len_hole_target, False)

    batch_subword_loss = tf.reduce_mean(tf.reduce_sum(masked_loss, axis=1)/ tf.cast(seq_len_hole_target, dtype=tf.float32))
    token_loss = batch_token_loss.numpy()

    total_subword_loss += batch_subword_loss.numpy()
    total_token_loss += token_loss
    token_losses.append(token_loss)

    hole_features[hole_identity.numpy()]=token_loss
    total_batches += 1

    if total_batches%10==0:
        bar.update(10)
        postfix = OrderedDict(batch_loss={batch_token_loss.experimental_ref()})
        bar.set_postfix(postfix)

  # Calculate mean batch_wise losses
  subword_loss = total_subword_loss/ total_batches
  token_loss = total_token_loss/ total_batches
  # confidence interval error
  error = Z * np.sqrt(np.var(token_losses)/total_batches)
  return subword_loss, token_loss, error, hole_features


def main():

  args = setup_args()

  #Create Models and Outputs directory if it doesn't exist
  Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
  Path(args.out_dir).mkdir(parents=True, exist_ok=True)

  outfile = args.out_dir + args.comment
  f_out = open(outfile, 'w')

  #Write meta-info about the particular run into the master file before each run
  timestr = time.strftime("%Y%m%d-%H%M%S")
  f = open(master_meta_info_file, 'a+')
  f.write(timestr+" #### "+ args.comment+ " ##### " + str(args)+"\n")
  f.close()

  hole_feature_filename = args.out_dir + "hole_features_"+ args.comment

  dataset = getData(args.hole_window_size, args.num_files*args.num_of_holes_per_file, args.dataset_type, args.sup_window_size, args.num_sup_tokens, args.num_of_holes_per_file, args.sup_def, args.method)

  #Get the size of the vocabulary
  vocab_size, encoder = get_vocab_size()

  model = Seq2SeqModel(vocab_size, bias_init=None)

  if args.load_model:
      y = tf.reshape(tf.Variable(1, dtype=tf.int32), (1,1))
      model(y, y, False)
      model.load_weights(args.model_load_dir).expect_partial() #to supress warnings
      print("Loaded Weights from: ", args.model_load_dir)

  size = args.num_files*args.num_of_holes_per_file
  bar = tqdm(total=size)

  print("Evaluating " + args.dataset_type +" Data.......")

  subword_loss, token_loss, error, hole_features = evaluate(model, dataset, args.method, bar, args.inner_learning_rate, args.sup_batch_size, args.num_of_updates)

  bar.close()
  print(args.dataset_type + " Statistics..........")
  f_out.write(args.dataset_type + " Statistics..........")

  print("Token Cross-Entropy = {:.4f} ".format(token_loss))
  print("{:.4f} confidence error over mean cross-entropy = {:.4f}".format(CONFIDENCE_INTERVAL, error))

  f_out.write("Token Cross-Entropy = {:.4f} ".format(token_loss))
  f_out.write("{:.4f} confidence error over mean cross-entropy = {:.4f}".format(CONFIDENCE_INTERVAL, error))
  f_out.flush()
  with open(hole_feature_filename, 'wb') as f:
      pickle.dump(hole_features, f)


if __name__ == "__main__":
  main()
