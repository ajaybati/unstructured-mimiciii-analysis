# BERT Search

BERT search is a transfer learning project that uses BERT's word embeddings to conduct a fuzzy search for similar Diseases of Despair patients in the MIMIC-III dataset. There are two main folders for this project: data_prep and BERT, where data is explored and manipulated and it is used as training data to feed our model, BERT. 

## Requirements

It is recommended you create a virtual environment. This project was done using the conda environment and anytime you need to run anything you must activate this environment. Look towards the [conda cheat sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf) for more information.

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install all of the required packages (all of them are available in [requirements.txt](requirements.txt)). 

For Jupyter notebooks, you may need to use pip install (module name) in one of the cells and restart the kernel for it to update. All the modules required for each file are listed in the first cell.

## Usage

### data_prep

This work is based on the data tables available in the [MIMIC-III dataset](https://mimic.physionet.org/gettingstarted/access/). Most of the first file is for data analysis and setting up our training data. The first few cells involve creating the diseases of despair codes, so it can be compiled in one database to access the notes of each of the patients. If there are any other diagnoses, it would need too be added in number_slot with the number of ICD9 codes in front of it (6 in this case). Some codes have 'E' in front of it, so we set addE to True.

```python
flat_filtered_list.append(getIndex("number_slot",6, addE=True))
```

Then, use the merging code to get the database for the notes. This is the major part of our training data and notes_df2 contains all the notes that are used as training data. The next couple of steps involve more data analysis to examine the top occurrences of words in every note. The cells after this point are split into 'general' and 'DOD'. Run 3 nested functions for both 'general' and 'DOD' to get the overall occurrence of a word, the occurence of words unique in each note, and the occurence of words in each patient. 

```python
create_accum()
second_accum()
third_accum()
```

Then, to get the TF-IDF it is passed through a function to get a list of this for both general and DOD notes. The activation functions are to get the top 80% of TF-IDF, so the most weighted words can be examined and analyzed. At the end, there should be words relating to respiratory issues, liver diseases, and medications and procedures relating to liver and respiratory system.

MAKE SURE to save each of the files as pickles, so accessing them takes a short amount of time. In datanalysis, there are comments on some cells (as seen below), stating that after running through each cell, the order to retrieve all information back is by running the selected cells.

```python
#while loading=(number)
```

### BERT

The training data is assembled in BERT.ipynb. There are 4 required files: input_ids with masks, attenntion masks, input_ids without masks, and mask indices. The first three files can be created using textLoader class as seen in bertmodelDS.py (if you have access to a gpu, it would helpful since running on a cpu takes a long time)

```python
data_2 = textLoader(yes = True)

data_input_randomized = torch.cat(tuple(data_2.input_ids_all),dim = 0)
attention_masks_randomized = torch.cat(tuple(data_2.attention_masks_all),dim = 0)
print("save 1")
torch.save(data_input_randomized,path+"DSdata128/DSmodeldata128.pickle")
print("save 2")
torch.save(attention_masks_randomized,path+"DSdata128/DSmodeldataAM128.pickle")
print("done")

print("="*20+" start "+"="*20)
data_real = textLoader(yes = False)

data_input = torch.cat(tuple(data_real.input_ids_all),dim = 0)
attention_masks = torch.cat(tuple(data_real.attention_masks_all),dim = 0)
print("save 1")
torch.save(data_input,path+"DSdata128/DSmodeldatareal128.pickle")
print("save 2")
torch.save(attention_masks,path+"DSdata128/DSmodeldataAMreal128.pickle")
print("done")
```

Then, using the data_input_randomized, create the mask_ids_tensor. The script to do so can be found both in bermodelDS.py and BERT.ipynb.

The later cells are experiments done using the pretrained BERT model (before running our training algorithm). If you want to know more about the functions available with BERT check out [huggingface BERT MLM](https://huggingface.co/transformers/model_doc/bert.html).

The functions for calculating accuracy, bleu score, and other helper functions are available in the training loop in BERTrain-Copy2.py.

Run the training loop as is (gpu highly recommended) and it should take around 20-24 hours. You may need to split the training into segments, so save the model at each segment and resume training. 


### Work in progress: Search Algorithm

We are still developing efficient methods to build our search algorithm using the internal representations of BERT. Some of our tests are available in train_output.ipynb or search.py. 

The search.py searches through the UMLS metathesaurus and generates a lexicon through the boolean function is_SUICIDE().


