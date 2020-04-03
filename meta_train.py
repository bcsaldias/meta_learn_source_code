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
Description: Script for meta-training using reptile or fomaml. Trains for one epoch followed by evaluating on the validation data to determine when to stop
'''

#base directory path
base_dir = './'

#base data directory path
base_data_dir = os.path.join(base_dir, 'Preprocessed_Data')

#path of the subword dict file with counts which is used to initialize the biases of the dense layer at decoder
subword_vocab_filename = os.path.join(base_data_dir, 'subword_vocab_counts.dict')

#the file which stores meta-info about each run of the code
master_meta_info_file = os.path.join(base_dir, 'runs.txt')


TOTAL_TRAIN_FILES = 12934
TOTAL_VAL_FILES = 7185

CONFIDENCE_INTERVAL = 0.95

def setup_args():
  """
  Description: Takes in the command-line arguments from user
  """
  parser = argparse.ArgumentParser()
  parser.add_argument("--comment", type=str, default='tssa-reptile-train', help="comment used to identify the run")
  parser.add_argument("--out_dir", type=str, default='Outputs/', help="Output Directory.")
  parser.add_argument("--model_load_dir", type=str, default='Trained_Models/base_model/', help="Directory from which the model is loaded")
  parser.add_argument("--checkpoint_dir", type=str, default='Models/', help="Directory for checkpoint.")
  parser.add_argument("--save_model", type=bool, default=True, help="Whether to save the current model")
  parser.add_argument("--load_model", type=bool, default=True, help="Whether to load a pretrained model")
  parser.add_argument("--hole_window_size", type=int, default=200, help="Size of the Context window around the hole")
  parser.add_argument("--sup_window_size", type=int, default=200, help="Size of the Context window around the support token")
  parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs of training")
  parser.add_argument("--num_train_files", type=str, default=12934, help="Number of files to be taken for training. Use 12934 for full  train data")
  parser.add_argument("--num_val_files", type=str, default=7185, help="Number of files to be taken for validation. use 7185 for full val data")
  parser.add_argument("--batch_size_sup", type=int, default=20, help="Batch size of the support set")
  parser.add_argument("--num_sup_tokens", type=int, default=1024, help="Number of support tokens to be taken")
  parser.add_argument("--inner_learning_rate", type=float, default=1e-5, help="Initial Outer Learning rate for Adam Optimizer")
  parser.add_argument("--outer_learning_rate", type=float, default=1e-5, help="Initial Inner Learning rate for Adam Optimizer")
  parser.add_argument("--epsilon_reptile", type=float, default=0.1, help="Epsilon for reptile outer step")
  parser.add_argument("--val_monitor_interval", type=int, default=3, help="Number of epochs it waits for the val loss to decrease")
  parser.add_argument("--train_method", type=str, default='fomaml', help="method of meta-training: fomaml, reptile")
  parser.add_argument("--num_of_holes_per_file", type=int, default=1, help="Number of hole targets to be sampled per file")
  parser.add_argument("--sup_def", type=str, default='vocab', help="Definition of support token to be used: vocab, proj, random, unique")
  parser.add_argument("--num_of_updates", type=int, default=14, help="Number of inner updates done per hole target (=k) for TSSA")

  return parser.parse_args()

def train(model, optimizer_inner, optimizer_outer, dataset, train_method, bar, epsilon_reptile, batch_size_sup, num_of_updates):
  """
  Description: Performs meta-training for one epoch
  """
  if CONFIDENCE_INTERVAL == 0.95:
      Z = 1.96
  elif CONFIDENCE_INTERVAL == 0.99:
      Z = 2.58
  token_losses = []

  #During training, we want different holes to be sampled from the same file across epochs. So we do not want a fixed seed
  np.random.seed(None)

  total_subword_loss = 0.0
  total_token_loss = 0.0
  total_batches = 0
  for (batch, (hole_window, hole_target, seq_len_hole_target, sup_window, sup_token, seq_len_sup_token, hole_identity, sup_flag)) in enumerate(dataset):

    if sup_flag:
      sup_window = tf.squeeze(sup_window, axis=0)
      sup_token = tf.squeeze(sup_token, axis=0)
      seq_len_sup_token = tf.squeeze(seq_len_sup_token, axis=0)

    #Storing weights for use in reptile later
    old_model_trainable_variables = []
    for entry in model.get_weights():
      old_model_trainable_variables.append(entry)

    # Get the new model instance after doing num_of_updates inner updates and then calculate the gradient of the hole loss w.r.t the updated parameters to give the outer update
    if sup_flag and train_method=='fomaml':
      model_new = losses.support_loss_train(model, sup_window, sup_token, seq_len_sup_token, True, optimizer_inner, batch_size_sup, num_of_updates)
      with tf.GradientTape() as g:
        batch_token_loss, masked_loss = losses.hole_loss(model_new, hole_window, hole_target, seq_len_hole_target, True)
      grads = g.gradient(batch_token_loss, model.trainable_variables)
      optimizer_outer.apply_gradients(losses.clip_gradients(zip(grads, model.trainable_variables)))

    # Get the new model instance after doing num_of_updates inner updates and then calculate the outer update of reptile
    if sup_flag and train_method=='reptile':
      model_new = losses.support_loss_train(model, sup_window, sup_token, seq_len_sup_token, True, optimizer_inner, batch_size_sup, num_of_updates)
      batch_token_loss, masked_loss = losses.hole_loss(model_new, hole_window, hole_target, seq_len_hole_target, True)
      new_weights = []
      for i in range(len(model_new.trainable_variables)):
        new_weights.append(old_model_trainable_variables[i] + epsilon_reptile*(model_new.trainable_variables[i]-old_model_trainable_variables[i]))
      model.set_weights(new_weights)

    # If there are no support tokens found in the file directly calculate the gradient of the hole target loss
    if not sup_flag:
      with tf.GradientTape() as g:
        batch_token_loss, masked_loss = losses.hole_loss(model, hole_window, hole_target, seq_len_hole_target, True)
      grads = g.gradient(batch_token_loss, model.trainable_variables)
      optimizer_outer.apply_gradients(losses.clip_gradients(zip(grads, model.trainable_variables)))

    batch_subword_loss = tf.reduce_mean(tf.reduce_sum(masked_loss, axis=1)/ tf.cast(seq_len_hole_target, dtype=tf.float32))
    token_loss = batch_token_loss.numpy()
    total_subword_loss += batch_subword_loss.numpy()
    total_token_loss += token_loss
    token_losses.append(token_loss)
    total_batches += 1

    if total_batches % 10 == 0:
        bar.update(10)
        postfix = OrderedDict(batch_loss = {token_loss})
        bar.set_postfix(postfix)

  # Calculate mean batch_wise losses
  subword_loss = total_subword_loss/ total_batches
  token_loss = total_token_loss/ total_batches
   # confidence interval error
  error = Z*np.sqrt(np.var(token_losses)/total_batches)
  return subword_loss, token_loss, error


def evaluate(model, dataset, bar, inner_learning_rate, sup_batch_size, num_of_updates):
  """
  Description: Performs evaluation for one epoch. Both the optimizer state and model parameters are reset before calculating the next hole target cross-entropy.
  """
  if CONFIDENCE_INTERVAL == 0.95:
      Z = 1.96
  elif CONFIDENCE_INTERVAL == 0.99:
      Z = 2.58

  np.random.seed(42)
  token_losses = []
  hole_features = {}

  total_subword_loss = 0.0
  total_token_loss = 0.0
  total_batches = 0

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

      model_new = losses.inner_loss_eval(model, sup_window, sup_token, seq_len_sup_token, False, 'tssa', inner_learning_rate, sup_batch_size, num_of_updates)
      batch_token_loss, masked_loss = losses.hole_loss(model_new, hole_window, hole_target, seq_len_hole_target, False)

    # If there are no support tokens found in the file directly calculate the hole target loss using the hole window
    if not sup_flag:
      batch_token_loss, masked_loss = losses.hole_loss(model, hole_window, hole_target, seq_len_hole_target, False)

    batch_subword_loss = tf.reduce_mean(tf.reduce_sum(masked_loss, axis=1)/ tf.cast(seq_len_hole_target, dtype=tf.float32))
    token_loss = batch_token_loss.numpy()

    total_subword_loss += batch_subword_loss.numpy()
    total_token_loss += token_loss
    token_losses.append(token_loss)

    hole_features[hole_identity.numpy()]=token_loss
    total_batches += 1

    if total_batches % 10 == 0:
        bar.update(10)
        postfix = OrderedDict(batch_loss = {token_loss})
        bar.set_postfix(postfix)

  #For training for next epoch
  model.set_weights(trained_model_trainable_variables)

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


  dataset_train = getData(args.hole_window_size, args.num_train_files*args.num_of_holes_per_file, 'train', args.sup_window_size, args.num_sup_tokens, args.num_of_holes_per_file,
                          args.sup_def, is_eval=False, data_type='hole_and_sup')

  dataset_val = getData(args.hole_window_size, args.num_val_files*args.num_of_holes_per_file, 'val', args.sup_window_size, args.num_sup_tokens, args.num_of_holes_per_file,
                          args.sup_def, is_eval=True, data_type='hole_and_sup')

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

  #Define the optimizers and initialize the seq2seq model
  optimizer_outer = tf.compat.v1.train.AdamOptimizer(learning_rate = args.outer_learning_rate)
  optimizer_inner = tf.compat.v1.train.AdamOptimizer(learning_rate = args.inner_learning_rate)
  model = Seq2SeqModel(vocab_size, bias_init)

  #the model save directory is named based on the comment (so that it is easy to restore it if needed)
  model_save_dir = args.checkpoint_dir + args.comment +"/"
  if not os.path.exists(model_save_dir):
      os.mkdir(model_save_dir)

  if args.load_model:
      y = tf.reshape(tf.Variable(1, dtype=tf.int32), (1,1))
      model(y, y, False)
      model.load_weights(args.model_load_dir)
      print("Loaded Weights from: ", args.model_load_dir)

  # required for tqdm bar progress
  train_size = args.num_train_files*args.num_of_holes_per_file

  #calculate initial perplexity over the training data (without training)
  tqdm.write('Calculating Initial Train Cross-Entropy')
  bar = tqdm(total=train_size)

  init_subword_loss, init_token_loss, init_train_error, hole_features = evaluate(model, dataset_train, bar, args.inner_learning_rate, args.batch_size_sup, args.num_of_updates)

  print("Init Token Train Cross-Entropy = {:.4f} ".format(init_token_loss))
  print("{:.4f} confidence error over init train mean cross-entropy = {:.4f}".format(CONFIDENCE_INTERVAL, init_train_error))

  f_out.write("Init Token Train Cross-Entropy = {:.4f}  ".format(init_token_loss)+"\n")
  f_out.write("{:.4f} confidence error over init train mean cross-entropy = {:.4f}".format(CONFIDENCE_INTERVAL, init_train_error)+"\n\n")

  # required for tqdm bar progress
  val_size = args.num_val_files*args.num_of_holes_per_file

  best_token_loss = None
  val_losses = []

  for epoch in range(args.num_epochs):

    hole_feature_filename = args.out_dir + "epoch_"+str(epoch+1)+"_hole_features_"+ args.comment

    #Train 1 epoch
    bar = tqdm(total=train_size)
    # calculate epoch wise train cross-entropy
    outfile = args.out_dir + args.comment
    train_subword_loss, train_token_loss, error = train(model, optimizer_inner, optimizer_outer, dataset_train, args.train_method, bar, args.epsilon_reptile, args.batch_size_sup, args.num_of_updates)
    bar.close()

    #Evaluate 1 epoch
    bar = tqdm(total=val_size)
    # calculate val cross-entropy
    val_subword_loss, val_token_loss, val_error, hole_features = evaluate(model, dataset_val, bar, args.inner_learning_rate, args.batch_size_sup, args.num_of_updates)
    val_losses.append(val_token_loss)
    bar.close()

    with open(hole_feature_filename, 'wb') as f:
      pickle.dump(hole_features, f)

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
    print('\n{:.4f} confidence error over mean train_train cross-entropy = {:.4f}, mean val cross-entropy = {:.4f}'.format(CONFIDENCE_INTERVAL, error, val_error)+"\n")

    f_out.write('Epoch {}: Train Token Cross-Entropy = {:.4f}, Val Token Cross-Entropy = {:.4f}'.format(epoch + 1, train_token_loss, val_token_loss)+"\n")
    f_out.write('{:.4f} confidence error over mean train_train cross-entropy = {:.4f}, mean val cross-entropy = {:.4f}'.format(CONFIDENCE_INTERVAL, error, val_error)+'\n\n')
    f_out.flush()

  #print best performance on val data over all epochs
  print("\n Best Token Val Cross-Entropy = {:.4f} ".format(best_token_loss))
  f_out.write("Best Token Val Cross-Entropy = {:.4f} ".format(best_token_loss)+"\n")
  f_out.close()

if __name__ == "__main__":
  main()
