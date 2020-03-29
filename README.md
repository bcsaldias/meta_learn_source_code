## On-the-Fly Adaptation of Source Code Models using Meta-Learning
Disha Shrivastava, Hugo Larochelle, Daniel Tarlow

This repository contains implementation and data for our work [On-the-Fly Adaptation of Source Code Models using Meta-Learning](). A block diagram of our approach can be found below. For more details, refer to the paper.

<p align="center">
 <img src="block_diagram.png" width=600>
</p>

## Dependencies
* python 3.7
* tensorflow-gpu 2.0.0 or tf-nightly-gpu-2.0-preview
* tensor2tensor
* tqdm
* javac-parser

## Data
* Download the full Java Github Corpus from [here](http://groups.inf.ed.ac.uk/cup/javaGithub/) (java_projects.tar.gz). Extract the data and place in the Raw_Data folder.
* Obtain the list of projects in 1% corpus of train, test and validation splits of the data from [here](https://github.com/SLP-team/SLP-Core/tree/master/FSE'17%20Replication) (*-projects.txt files). Create folders named 'train', 'val' and 'test' which contain these projects from the java_projects folder obtained in the step above. Place these folders in Raw_Data directory.
* Download the jar from [here](https://github.com/SLP-team/SLP-Core/releases/download/v0.2/SLP-Core_v0.2.jar) (SLP-Core_v0.2.jar). Place it in the Raw_Data directory. To lex the corpus, run `java -jar SLP-Core_v0.2.jar lex x x-lexed -l java` where, x = train, test, val (requires a java installation). x should point to the 'train', 'val' and 'test' folders formed in the previous step. After lexing, you will see .java files inside x-lexed folders with comments removed and java tokenized text separted by tabs.
* Run extract_data.py (Steps 2, 3, 5, 6). This will result in formation of data_x.txt and data_x.json files with x = train, test, val in the Preprocessed_Data directory.
* Run preprocess_data.py. This will generate files basic_dict.x and episodes_x.csv where x = train, test, val

## Repository Structure
- Models : Directory for storing the models (will be created)
- Outputs : Directory for storing the outputs (output runs as well as hole features) (will be created)
- Trained_Models (can be downloaded from [here](https://drive.google.com/file/d/1fzJP5qejRfVpxRAEKTY1BiyhfNC6O00T/view?usp=sharing))
	- base_model : Trained base model
	- tssa_fomaml : TSSA-FOMAML best model
	- tssa_reptile : TSSA-Reptile best model
- Preprocessed_Data
	- subword_vocab.txt : Subword vocab
	- subword_vocab_counts.dict : Subword vocab with counts
	- token_vocab.dict : token vocab with counts
- Raw_Data
    - not_vocab_1_percent.txt : list of projects in the train, test and val splits of the 1% corpus and hence not to be included while forming the vocab split
- data.py : Creates data iterators
- model.py : Model definition and call functions
- losses.py : Loss functions
- generate_episodes.py : Creates episodes consisting of hole target and coreesponding support tokens
- test.py : Evaluation script
- train_base_model.py : Training of base model
- meta_train.py : Training with TSSA-FOMAML or TSSA-Reptile
- extract_data.py : To extract 1% corpus from raw data and generate json and text files
- preprocess_data.py : To preprocess data
- runs.txt : Stores meta-info corresponding to each run

## Replicating results
The trained models can be downloaded from [here](https://drive.google.com/file/d/1fzJP5qejRfVpxRAEKTY1BiyhfNC6O00T/view?usp=sharing) (Place it in the root folder).
To replicate results in Table-2 of the paper, run the commands below:
 * Base Model: `python test.py --method base_model --comment test_base_model`
 * Dynamic Evaluation: `python test.py --method dyn_eval --inner_learning_rate 1e-3 --comment test_dyn_eval`
 * TSSA-1: `python test.py --method tssa --inner_learning_rate 5e-3 --num_of_updates 1 --sup_def proj --num_sup_tokens x --sup_batch_size x --comment test_tssa_1_x (where x = 256, 512, 1024)`
 * TSSA-k: `python test.py --method tssa --inner_learning_rate 5e-4 --num_of_updates 16 --num_sup_tokens x --comment test_tssa_k_x (where x = 256, 512, 1024)`
 * TSSA-Reptile: `python test.py --method tssa --inner_learning_rate 5e-4 --num_of_updates 16 --num_sup_tokens x --model_load_dir 'Trained_Models/tssa_reptile/' --comment test_tssa_reptile_x (where x = 256, 512, 1024)`
 * TSSA-FOMAML: `python test.py --method tssa --inner_learning_rate 5e-4 --num_of_updates 16 --num_sup_tokens x --model_load_dir 'Trained_Models/tssa_fomaml/' --comment test_tssa_fomaml_x (where x = 256, 512, 1024)`

To train the base model run: `python train_base_model.py` with default parameters

To meta-train with Reptile : `python meta_train.py --train_method reptile --num_sup_tokens 512 --num_of_updates 32 --sup_def proj --inner_learning_rate 5e-5 --checkpoint_dir Models/tssa_reptile --comment train_val_tssa_reptile`

To meta-train with FOMAML : `python meta_train.py --train_method reptile --num_sup_tokens 1024 --num_of_updates 14 --checkpoint_dir Models/tssa_fomaml --comment train_val_tssa_fomaml`

Note: If you are training your own base model, it is better to initilize the meta-training with the trained base model to get faster convergence (this happens by deafult currently in the code)

Disclaimer: In some versions of tf-nightly-gpu, you might get an error regarding the use of experimental_ref() for tqdm progress bar. In those cases just remove experimental_ref() and the script should run fine.

## Citation

If you use our code, please consider citing us as below:

```
@misc{shrivastava2020onthefly,
    title={On-the-Fly Adaptation of Source Code Models using Meta-Learning},
    author={Disha Shrivastava and Hugo Larochelle and Daniel Tarlow},
    year={2020},
    eprint={2003.11768},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}

```
