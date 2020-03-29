import os
import csv
import json
import pickle
import random
from tensor2tensor.data_generators import text_encoder

'''
Description: Creates a pickle file with each entry in the dict indexed by proj_id, dir_id, file_id and the key contains contents of the full file.
Creates a csv file for use in data iterator where each row of the csv consists of one file
'''
#base directory path
base_dir = './'

base_data_dir = os.path.join(base_dir, 'Preprocessed_Data')

subword_vocab_filename = os.path.join(base_data_dir, 'subword_vocab.txt')
token_dict_filename = os.path.join(base_data_dir, 'token_vocab.dict')

end_of_token_index = 1 # index in the subtoken vocab of <EOT>
epsilon = 1e-5
random.seed(42)
encoder = text_encoder.SubwordTextEncoder(subword_vocab_filename)
token_dict = pickle.load(open(token_dict_filename, 'rb'))

def get_dir_dict(data, proj_id, dir_id):
  dir_dict = {}
  for j in range(len(data[proj_id][dir_id])):
    file_subtoken_string, file_dict = data[proj_id][dir_id][j]
    if file_subtoken_string:
      for ent in file_dict:
        token = file_dict[ent][0]
        #Count frequency in project
        if token in dir_dict:
          dir_dict[token]+= 1.0
        else:
          dir_dict[token] = 1.0
  return dir_dict

def create_file_DS(data_json_filename, ds_filename):
  basic_DS = {}
  data = json.loads(open(data_json_filename, encoding='utf-8', errors='backslashreplace').read())
  for i in range(len(data)):
    print("Project: ", str(i))
    proj_id = int(data[i]['project_index'])
    directories = data[i]['directories']
    basic_DS[proj_id] = {}
    for j in range(len(directories)):
      dir_id = int(directories[j]['directory_index'])
      basic_DS[proj_id][dir_id] = {}
      files = directories[j]['files']
      for k in range(len(files)):
        doc_string = ''
        file_subword_indices = []
        end_of_line_indices = []
        token_subword = {}
        file_dict = {}
        file_id = int(files[k]['file_index'])
        lines = files[k]['lines']
        for l in range(len(lines)):
          tokens = lines[l]['tokens'].strip().split() #For 1% corpus formed by using jsondumps
          token_types = lines[l]['token_types'].strip().split()
          subword_id_ranges = []
          entries = 0
          for t in range(len(token_types)):
            token = tokens[t]
            token_type = token_types[t]
            if token_type == 'STRINGLITERAL':
              for tt in range(t, len(tokens)):
                if tokens[tt].endswith("\"") or tokens[tt].endswith("\'"):
                  token = ' '.join(tokens[t:tt+1])
                  break
            init_length = len(file_subword_indices)
            subword_tokens = encoder.encode_without_tokenizing(token)
            if len(subword_tokens)> 20:
              continue
            subword_id_range = (init_length,init_length+len(subword_tokens)+1)
            if subword_id_range not in file_dict:
              file_dict[subword_id_range] = []
            if token in token_dict:
              file_dict[subword_id_range].append(token)
              file_dict[subword_id_range].append(token_types[t])
              file_dict[subword_id_range].append(token_dict[token])
            else:
              file_dict[subword_id_range].append(token)
              file_dict[subword_id_range].append(token_types[t])
              file_dict[subword_id_range].append(0.0)

            for s in subword_tokens:
              file_subword_indices.append(s)

            file_subword_indices.append(end_of_token_index)
            entries+=1
            subword_id_ranges.append(subword_id_range)

          for entry in subword_id_ranges:
              file_dict[entry].append(len(file_subword_indices))

        basic_DS[proj_id][dir_id][file_id] = (file_subword_indices, file_dict)

  with open(ds_filename, 'wb') as f:
  	pickle.dump(basic_DS, f)


def store_file_episodes(ds_filename, csv_filename):
  data = pickle.load(open(ds_filename, 'rb'))
  episodes_list = []
  for i in range(len(data)):
    print("Project: ", str(i))
    for j in range(len(data[i])):
      dir_dict = get_dir_dict(data, i, j)
      for k in range(len(data[i][j])):
        file_target_holes = []
        file_subtoken_string, file_dict = data[i][j][k]
        if file_subtoken_string:
          for ent in file_dict:
            subtoken_id_range = ent
            if subtoken_id_range[1] - subtoken_id_range[0] > 20:
              continue
            token = file_dict[ent][0]
            token_type = file_dict[ent][1]
            vocab_count = file_dict[ent][2]
            end_of_line_index = file_dict[ent][3]
            dir_count = dir_dict[token]
            if vocab_count ==0.0:
              vocab_count = epsilon
            hole_str = str(subtoken_id_range[0])+'%%'+str(subtoken_id_range[1]) +'%%'\
                                            +str(end_of_line_index) +'%%'+token_type+'%%'\
                                            +str(vocab_count)+'%%'+str(dir_count)
            file_target_holes.append(hole_str) #position

        episodes_list.append([i, j, k, file_subtoken_string, file_target_holes])

  random.shuffle(episodes_list)
  with open(csv_filename, 'w', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    for entry in episodes_list:
      writer.writerow(entry)

if __name__== "__main__":

  #Write train basic_dict data and episodes csv file
  ds_filename = os.path.join(base_data_dir, 'basic_DS_dict.train')
  data_json_filename = os.path.join(base_data_dir, 'data_train.json')
  csv_filename = os.path.join(base_data_dir, 'episodes_train.csv')
  print("Preprocessing train episodes...")
  create_file_DS(data_json_filename, ds_filename)
  print("Writing train episodes...")
  store_file_episodes(ds_filename, csv_filename)

  #Write val basic_dict data and episodes csv file
  ds_filename = os.path.join(base_data_dir, 'basic_DS_dict.val')
  data_json_filename = os.path.join(base_data_dir, 'data_val.json')
  csv_filename = os.path.join(base_data_dir, 'episodes_val.csv')
  print("Preprocessing val episodes...")
  create_file_DS(data_json_filename, ds_filename)
  print("Writing val episodes...")
  store_file_episodes(ds_filename, csv_filename)


  #Write test basic_dict data and episodes csv file
  ds_filename = os.path.join(base_data_dir, 'basic_DS_dict.test')
  data_json_filename = os.path.join(base_data_dir, 'data_test.json')
  csv_filename = os.path.join(base_data_dir, 'episodes_test.csv')
  print("Preprocessing test episodes...")
  create_file_DS(data_json_filename, ds_filename)
  print("Writing test episodes...")
  store_file_episodes(ds_filename, csv_filename)

