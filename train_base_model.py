import os
import sys
import time
import pickle
import argparse
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict


#Importing from other programs
import losses
from model import Seq2SeqModel
from data import getData, get_vocab_size

os.environ["CUDA_VISIBLE_DEVICES"]="0"

'''
Description: Script for training the base model
'''

#base directory path
base_dir = './'

#base data directory path
base_data_dir = os.path.join(base_dir, 'Preprocessed_Data')

#path of the subword dict file used to initialize the biases of the dense layer at decoder
subword_vocab_filename = os.path.join(base_data_dir, 'subword_vocab_counts.dict')

#the file which stores meta-info about each run of the code
master_meta_info_file = os.path.join(base_dir, 'runs.txt')

TOTAL_TRAIN_TOKENS = 15656538
TOTAL_VAL_TOKENS  = 3813584

CONFIDENCE_INTERVAL = 0.95

def setup_args():
  """
  Description: Takes in the command-line arguments from user
  """
  parser = argparse.ArgumentParser()

  parser.add_argument("--comment", type=str, default='base_model_train', help="comment used to identify the run")
  parser.add_argument("--out_dir", type=str, default='Outputs/', help="Output Directory.")
  parser.add_argument("--checkpoint_dir", type=str, default='Models/', help="Directory for checkpoint.")
  parser.add_argument("--save_model", type=bool, default=True, help="Whether to save the current model")
  parser.add_argument("--hole_window_size", type=int, default=200, help="Size of the Context window around the hole target")
  parser.add_argument("--batch_size_hole", type=int, default=512, help="Batch size")
  parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs of training")
  parser.add_argument("--num_train_examples", type=str, default='full', help="Number of examples to be taken for training. Use 'full' for full train data")
  parser.add_argument("--num_val_examples", type=str, default='full', help="Number of examples to be taken for validation. Use 'full' for full val data")
  parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial Learning rate for Adam Optimizer")
  parser.add_argument("--val_monitor_interval", type=int, default=3, help="Number of epochs it waits for the val loss to decrease")

  return parser.parse_args()

def train(model, optimizer, dataset, bar):
  """
  Description: Performs training for one epoch
  """
  if CONFIDENCE_INTERVAL == 0.95:
      Z = 1.96
  elif CONFIDENCE_INTERVAL == 0.99:
      Z = 2.58

  total_subword_loss = 0.0
  total_token_loss = 0.0
  total_batches = 0
  token_losses = []

  for (batch, (hole_window, hole_target, seq_len_hole_target)) in enumerate(dataset):

    hole_window = tf.squeeze(hole_window, axis=1)
    hole_target = tf.squeeze(hole_target, axis=1)
    seq_len_hole_target = tf.squeeze(seq_len_hole_target, axis=1)

    with tf.GradientTape() as g:
      batch_token_loss, masked_loss = losses.hole_loss(model, hole_window, hole_target, seq_len_hole_target, True)

    grads = g.gradient(batch_token_loss, model.trainable_variables)
    optimizer.apply_gradients(losses.clip_gradients(zip(grads, model.trainable_variables)))

    batch_subword_loss = tf.reduce_mean(tf.reduce_sum(masked_loss, axis=1)/ tf.cast(seq_len_hole_target, dtype=tf.float32))
    total_subword_loss += batch_subword_loss.numpy()
    total_token_loss += batch_token_loss.numpy()
    token_losses.append(batch_token_loss.numpy())
    total_batches += 1

    if total_batches % 10 == 0:
        bar.update(10)
        postfix = OrderedDict(batch_loss = {batch_token_loss})
        bar.set_postfix(postfix)

  # Calculate mean batch_wise losses
  subword_loss = total_subword_loss/ total_batches
  token_loss = total_token_loss/ total_batches
  subword_ppl = np.exp(subword_loss)
  token_ppl = np.exp(token_loss)
  # confidence interval error
  error = Z* np.sqrt(np.var(token_losses)/total_batches)
  return subword_loss, token_loss, error


def evaluate(model, dataset, bar):
  """
  Description: Performs evaluation for one epoch
  """
  if CONFIDENCE_INTERVAL == 0.95:
      Z = 1.96
  elif CONFIDENCE_INTERVAL == 0.99:
      Z = 2.58

  total_subword_loss = 0.0
  total_token_loss = 0.0
  total_batches = 0
  token_losses = []

  for (batch, (hole_window, hole_target, seq_len_hole_target)) in enumerate(dataset):

    hole_window = tf.squeeze(hole_window, axis=1)
    hole_target = tf.squeeze(hole_target, axis=1)
    seq_len_hole_target = tf.squeeze(seq_len_hole_target, axis=1)

    batch_token_loss, masked_loss = losses.hole_loss(model, hole_window, hole_target, seq_len_hole_target, False)

    batch_subword_loss = tf.reduce_mean(tf.reduce_sum(masked_loss, axis=1)/ tf.cast(seq_len_hole_target, dtype=tf.float32))
    total_subword_loss += batch_subword_loss.numpy()
    total_token_loss += batch_token_loss.numpy()
    token_losses.append(batch_token_loss.numpy())
    total_batches += 1

    if total_batches % 10 == 0:
        bar.update(10)
        postfix = OrderedDict(batch_loss = {batch_token_loss.numpy()})
        bar.set_postfix(postfix)

  # Calculate mean batch_wise losses
  subword_loss = total_subword_loss/ total_batches
  token_loss = total_token_loss/ total_batches
  # confidence interval error
  error = Z * np.sqrt(np.var(token_losses)/total_batches)
  return subword_loss, token_loss, error


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

  dataset_train = getData(args.hole_window_size, args.num_train_examples, 'train', data_type='hole', hole_batch_size=args.batch_size_hole)

  dataset_val = getData(args.hole_window_size, args.num_val_examples, 'val', data_type='hole', hole_batch_size=args.batch_size_hole)

  #Get the size of the vocabulary
  vocab_size, encoder = get_vocab_size()

  # define bias initializer based on log(prob in vocab)
  subword_dict = pickle.load(open(subword_vocab_filename, 'rb'))
  subword_dict = {k: v / total for total in (sum(subword_dict.values()),) for k, v in subword_dict.items()}
  lowest_entry = min(subword_dict, key=subword_dict.get)
  subword_vocab = np.zeros(vocab_size)
  for i in range(vocab_size):
      if i in subword_dict:
          subword_vocab[i] = np.log(subword_dict[i])
      else:
          subword_vocab[i] = np.log(subword_dict[lowest_entry])


  def bias_init(shape, dtype=None, partition_info=None):
      return tf.convert_to_tensor(subword_vocab, dtype=tf.float32)

  #Define the optimizer and initialize the seq2seq model
  optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = args.learning_rate)
  model = Seq2SeqModel(vocab_size, bias_init)

  #the model save directory is named based on the comment (so that it is easy to restore it if needed)
  model_save_dir = args.checkpoint_dir + args.comment +"/"
  if not os.path.exists(model_save_dir):
      os.mkdir(model_save_dir)

  # if args.load_model:
  #     y = tf.reshape(tf.Variable(1, dtype=tf.int32), (1,1))
  #     model(y, y, False)
  #     model.load_weights(args.model_load_dir)
  #     print("Loaded Weights from: ", args.model_load_dir)

  if args.num_train_examples =='full':
      train_size = int(TOTAL_TRAIN_TOKENS/args.batch_size_hole)
  else:
      train_size = args.num_train_examples

  #calculate initial perplexity over the training data (without training)
  tqdm.write('Calculating Initial Train Cross-Entropy')
  bar = tqdm(total=train_size)

  init_subword_loss, init_token_loss, init_train_error = evaluate(model, dataset_train, bar)

  print("Init Token Train Cross-Entropy = {:.4f} ".format(init_token_loss))
  print("{:.4f} confidence error over init train mean cross-entropy = {:.4f}".format(CONFIDENCE_INTERVAL, init_train_error))

  f_out.write("Init Token Train Cross-Entropy = {:.4f}  ".format(init_token_loss)+"\n")
  f_out.write("{:.4f} confidence error over init train mean cross-entropy = {:.4f}".format(CONFIDENCE_INTERVAL, init_train_error)+"\n\n")

  if args.num_val_examples =='full':
      val_size = int(TOTAL_VAL_TOKENS/args.batch_size_hole)
  else:
      val_size = args.num_val_examples

  best_token_loss = None
  val_losses = []

  for epoch in range(args.num_epochs):

    #Train 1 epoch
    bar = tqdm(total=train_size)
    # calculate epoch wise train cross-entropy
    outfile = args.out_dir + args.comment
    train_subword_loss, train_token_loss, error = train(model, optimizer, dataset_train, bar)

    bar.close()

    #Evaluate 1 epoch
    bar = tqdm(total=val_size)
    # calculate val cross-entropy
    val_subword_loss, val_token_loss, val_error = evaluate(model, dataset_val, bar)
    val_losses.append(val_token_loss)
    bar.close()

    # checkpoint if the val_loss decreases
    if not best_token_loss or val_token_loss < best_token_loss:
      if args.save_model:
        model.save_weights(model_save_dir, save_format='tf')
        print("\nSaved Weights to: ", model_save_dir)

      best_subword_loss = val_subword_loss
      best_token_loss = val_token_loss

    #early stopping if val_loss doesn't improve
    if len(val_losses) >= args.val_monitor_interval:
        if np.all(np.array(val_losses)> best_token_loss):
          print("\n Early Stopping because val loss didn't improve after ", args.val_monitor_interval, "intervals")
          break
        else:
          val_losses.pop(0)

    #print epoch-wise statistics
    print('\nEpoch {}: Train Token Cross-Entropy = {:.4f}, Val Token Cross-Entropy = {:.4f}'.format(epoch + 1, train_token_loss, val_token_loss))
    print('\n{:.4f} confidence error over mean train_train cross-entropy = {:.4f}, mean val cross-entropy = {:.4f}'.format(CONFIDENCE_INTERVAL, error, val_error)+'\n')

    f_out.write('Epoch {}: Train Token Cross-Entropy = {:.4f}, Val Token Cross-Entropy = {:.4f}'.format(epoch + 1, train_token_loss, val_token_loss)+"\n")
    f_out.write('{:.4f} confidence error over mean train_train cross-entropy = {:.4f}, mean val cross-entropy = {:.4f}'.format(CONFIDENCE_INTERVAL, error, val_error)+'\n\n')
    f_out.flush()

  #print best performance on val data over all epochs
  print("\n Best Token Val Loss = {:.4f} ".format(best_token_loss))
  f_out.write(" Best Token Val Loss = {:.4f} ".format(best_token_loss)+"\n")
  f_out.close()

if __name__ == "__main__":
  main()
