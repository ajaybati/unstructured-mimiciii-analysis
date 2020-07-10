import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import BertTokenizer, BertForMaskedLM
from nltk.tokenize import sent_tokenize
import nltk
import time
import pickle
from torch.utils.data import DataLoader
path = "/global/cscratch1/sd/ajaybati/pickles/"


# noteevents = pd.read_csv("/project/projectdirs/m1532/Projects_MVP/_datasets/mimiciii/NOTEEVENTS.csv")


# final_df = pd.read_pickle(path+"final_df.pickle")
# notes_df = noteevents[["SUBJECT_ID","HADM_ID","CHARTDATE","CHARTTIME","CATEGORY","TEXT"]]
# notes_df = final_df[["HADM_ID","SUBJECT_ID","DOB"]].drop_duplicates().merge(notes_df, on=["SUBJECT_ID","HADM_ID"], how="right")
# print("done1")
# notes = noteevents[["SUBJECT_ID","HADM_ID","CHARTDATE","CHARTTIME","CATEGORY","TEXT"]]
# notes_df = pd.merge(final_df[["SUBJECT_ID","HADM_ID","DOB"]],notes.drop_duplicates(subset = ["SUBJECT_ID","HADM_ID","CHARTDATE","CHARTTIME"]), on=["SUBJECT_ID","HADM_ID"], how = "left")
# notes_df = notes_df.drop_duplicates(subset='TEXT')
# print("done2")

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertForMaskedLM.from_pretrained('bert-base-uncased')


# import random
# class textLoader(Dataset):
#     def __init__(self,transform = None, yes = True):
#         self.input_ids_all = []
#         self.attention_masks_all = []
#         self.text = notes_df["TEXT"].tolist()
#         self.samples = len(self.text)
#         total = 0
#         x=0
#         track = 0 
#         for note in self.text[256971:]:
#             start = time.time()
#             try:
#                 for sent in sent_tokenize(note):

#                     encoded_dict = tokenizer.encode_plus(
#                         sent,                      # Sentence to encode.
#                         add_special_tokens = True, # Add '[CLS]' and '[SEP]'
#                         max_length = 64,
#                         truncation = True,# Pad & truncate all sentences.
#                         pad_to_max_length = True,
#                         return_attention_mask = True,   # Construct attn. masks.
#                         return_tensors = 'pt',     # Return pytorch tensors.
#                    )

#                     # Add the encoded sentence to the list.    
#                     input_ids = encoded_dict['input_ids']
#                     attention_masks = encoded_dict['attention_mask']

#                     if yes:
#                         input_ids_real = []
#                         for index,word in enumerate(input_ids[0]):
#                             if int(word) != 101 and int(word) != 102:
#                                 rando = random.random()
#                                 random.seed()
#                                 if rando < 0.2 and int(word)!=0:
#                                     input_ids_real.append(103)
#                                 else:
#                                     input_ids_real.append(int(word))
#                             else:
#                                 input_ids_real.append(int(word))
#                         input_ids_real = torch.tensor(input_ids_real).view(1,64)

#                         self.input_ids_all.append(input_ids_real)
#                         self.attention_masks_all.append(attention_masks)
#                     else:

#                         self.input_ids_all.append(input_ids)
#                         self.attention_masks_all.append(attention_masks)

#             except Exception as e:
#                 print(str(e), input_ids)
#             end = time.time()
#             if((self.text.index(note)+1)%10000==0):
#                 x+=1
#                 total+=(end-start)
#                 print((self.text.index(note)+1-256971)/367101+256971/367101, total/(x)*((367101-256971-10000*x)/3600))
            
       
            
#     def __getitem__(self,index):
#         return self.input_ids_all[index],self.attention_masks_all[index]
    
    
#     def __len__(self):
#         return self.samples

    
# print("="*20,"data randomized","="*20)

# data_2 = textLoader(yes = True)
# print('done')
# print("done")
# data_input_randomized = torch.cat(tuple(data_2.input_ids_all),dim = 0)
# attention_masks_randomized = torch.cat(tuple(data_2.attention_masks_all),dim = 0)
# print("save 1")
# torch.save(data_input_randomized,path+"REDOinput_ids_randomized30.pickle")
# print("save 2")
# torch.save(attention_masks_randomized,path+"REDOattention_masks_randomized30.pickle")
# print("done")

print("="*20+" start "+"="*20)
input_ids_all = torch.load(path+"input_ids_randomized100.pickle")


mask_indices = []
x=0
for sent in input_ids_all:
    start=time.time()
    sentence = list(sent)
    mask = []
    for index,word in enumerate(sentence):
        if word==103:
            mask.append(index)
    mask_indices.append(mask)
    end = time.time()
    x+=1
    if x%500000==0:
        print("="*20+" "+str(x)+"/8286324"+"="*20)
        print(len(mask)/len(input_ids_all[x-1]))
        print((end-start)*(8286324-x)/3600)
        print("="*80)
print("="*20+" DONE!!! "+"="*20)
with open(path+"mask_indices.pickle","wb") as f:
    pickle.dump(mask_indices,f)