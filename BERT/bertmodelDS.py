#loading in training data and other required collections

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
print("done")

noteevents = pd.read_csv("/project/projectdirs/m1532/Projects_MVP/_datasets/mimiciii/NOTEEVENTS.csv")
print("done")

final_df = pd.read_pickle(path+"final_df.pickle")
notes_df = noteevents[["SUBJECT_ID","HADM_ID","CHARTDATE","CHARTTIME","CATEGORY","TEXT"]]
notes_df = final_df[["HADM_ID","SUBJECT_ID","DOB"]].drop_duplicates().merge(notes_df, on=["SUBJECT_ID","HADM_ID"], how="right")

notes = noteevents[["SUBJECT_ID","HADM_ID","CHARTDATE","CHARTTIME","CATEGORY","TEXT"]]
notes_df = pd.merge(final_df[["SUBJECT_ID","HADM_ID","DOB"]],notes.drop_duplicates(subset = ["SUBJECT_ID","HADM_ID","CHARTDATE","CHARTTIME"]), on=["SUBJECT_ID","HADM_ID"], how = "left")
notes_df = notes_df.drop_duplicates(subset='TEXT')
notes_df2 = notes_df[notes_df["CATEGORY"]=="Discharge summary"]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
import random
class textLoader(Dataset):
    def __init__(self,transform = None, yes = True):
        self.input_ids_all = []
        self.attention_masks_all = []
        self.text = notes_df2["TEXT"].tolist()
        length = len(self.text)
        print(length)
        self.samples = len(self.text)
        total = 0
        x=0
        track = 0 
        for note in self.text:
            start = time.time()
            try:
                for sent in sent_tokenize(note):

                    encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 128,
                        truncation = True,# Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )

                    # Add the encoded sentence to the list.    
                    input_ids = encoded_dict['input_ids']
                    attention_masks = encoded_dict['attention_mask']

                    if yes:
                        input_ids_real = []
                        for index,word in enumerate(input_ids[0]):
                            if int(word) != 101 and int(word) != 102:
                                rando = random.random()
                                random.seed()
                                if rando < 0.2 and int(word)!=0:
                                    input_ids_real.append(103)
                                else:
                                    input_ids_real.append(int(word))
                            else:
                                input_ids_real.append(int(word))
                        input_ids_real = torch.tensor(input_ids_real).view(1,128)

                        self.input_ids_all.append(input_ids_real)
                        self.attention_masks_all.append(attention_masks)
                    else:

                        self.input_ids_all.append(input_ids)
                        self.attention_masks_all.append(attention_masks)

            except Exception as e:
                print(str(e), input_ids)
            end = time.time()
            track+=1
            total+=(end-start)
            if(track%1000==0):
                x+=1
                total+=(end-start)
                print((self.text.index(note))/length,"     Time Remaining:      ",total/track*((length-x*1000)/3600))
            
       
            
    def __getitem__(self,index):
        return self.input_ids_all[index],self.attention_masks_all[index]
    
    
    def __len__(self):
        return self.samples

    
print("="*20,"data randomized","="*20)

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

data_reala = torch.load(path+'DSdata128/DSmodeldata128.pickle')


length = len(data_reala)
print(length)
print(len(data_reala[0]))
mask_indices = []
x=0
for sent in data_reala:
    start=time.time()
    sentence = list(sent)
    mask = []
    for index,word in enumerate(sentence):
        if word==103:
            mask.append(index)
    mask_indices.append(mask)
    end = time.time()
    x+=1
    if x%50000==0:
        print("="*20+" "+str(x)+"/"+str(length)+" "+"="*20)
        print("percent of masks:  ",len(mask)/(len(sent)-list(sent).count(torch.tensor([0])))*100,"%")
        print("Time remaining: ",(end-start)*(length-x)/3600)
        print("="*80)
print("="*20+" DONE!!! "+"="*20)
with open(path+"DSdata128/mask_indicesDS128.pickle","wb") as f:
    pickle.dump(mask_indices,f)


#getting max sentence length (for optimal token length)
maxdict = {}
x=0
total = 0
a=0
for note in notes_df2["TEXT"]:
    x+=1
    start = time.time()
    for sent in sent_tokenize(note):
        try:
            encdict = tokenizer.encode_plus(sent, return_tensors='pt')
            length = len(encdict['input_ids'][0])
            if length not in maxdict:
                maxdict[length] = 1
            else:
                maxdict[length]+=1
        except:
            pass
    end=time.time()
    total+=(end-start)
    if x%1000==0:
        a+=1
        print(a,"/",17)
        print(total/x*(len(notes_df2)-1000*a)/60)
        


with open(path+"countsentlength.pickle","wb") as f:
    pickle.dump(maxdict,f)

print("done")
max_df = pd.DataFrame(data = maxdict).set_index(maxdict.keys().sort())
max_df.to_pickle(path+"max_df.pickle")