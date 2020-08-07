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
torch.save(data_input_randomized,'path_to_data_with_masks')
print("save 2")
torch.save(attention_masks_randomized,'path_to_masks')
print("done")

print("="*20+" start "+"="*20)
data_real = textLoader(yes = False)

data_input = torch.cat(tuple(data_real.input_ids_all),dim = 0)
attention_masks = torch.cat(tuple(data_real.attention_masks_all),dim = 0)
print("save 1")
torch.save(data_input,'path_to_realdata')
print("save 2")
torch.save(attention_masks,"path_to_masks")
print("done")
```

Then, using the data_input_randomized, create the mask_ids_tensor. The script to do so can be found both in bertmodelDS.py and BERT.ipynb.

With this, the script in BERT.ipynb allows you to create pytroch datasets for both the training and validation portions:

```python
from torch.utils.data import TensorDataset, random_split

# Combine the training inputs into a TensorDataset.
dataset = TensorDataset(input_ids_all, attention_masks_all, input_ids_real, mask_indices_tensor)

# Create a 90-10 train-validation split.

# Calculate the number of samples to include in each set.
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

# Divide the dataset by randomly selecting samples.
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

# The DataLoader needs to know our batch size for training, so we specify it 
# here. For fine-tuning BERT on a specific task, the authors recommend a batch 
# size of 16 or 32.
batch_size = 1

# Create the DataLoaders for our training and validation sets.
# We'll take training samples in random order. 
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )
torch.save(train_dataloader,'path_to_train')
torch.save(validation_dataloader,'path_to_validation')
```


The later cells are experiments done using the pretrained BERT model (before running our training algorithm). If you want to know more about the functions available with BERT check out [huggingface BERT MLM](https://huggingface.co/transformers/model_doc/bert.html).

The functions for calculating accuracy, bleu score, and other helper functions are available in the training loop in BERTrain-Copy2.py.

Run the training loop as is (gpu highly recommended) and it should take around 20-24 hours. You may need to split the training into segments, so save the model's state dictionary, optimizer's state dictionary, epoch, training_stats, at each segment and resume training:

```python
torch.save({
            'epoch': epoch_i+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'total_train_loss': total_train_loss,
            'step': step_number,
            'training_stats':training_stats}, "path_to_saved_model")
```


### Work in progress: Search Algorithm

We are still developing efficient methods to build our search algorithm using the internal representations of BERT. Some of our tests are available in train_output.ipynb or search.py, search2.py, faisstest.py, and search3lexicon.py.

All searches attempt to compare a certain piece of a discharge summary to the text in the UMLS metathesaurus. If it passes a certain threshold, it is accepted as similar.

Search.py registers a forward hook to capture a certain layer of BERT and use its embeddings as a comparison to ottherr strings with cosine distance.

Search2.py attempts to use a pipeline with a DAG, directed acyclic graph and the pooling layer of BERT to compare strings.

Search3.py uses the pooler layer and other helper functions to build sentence features for strings to compare using cosine distance. 

Lastly, faisstest.py attempts to use Facebook's efficient similarity search to capture similar strings in UMLS.

