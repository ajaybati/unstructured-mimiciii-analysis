import os
import hy
import sys
sys.path.insert(0, '/global/project/projectdirs/m1532/Projects_MVP/_datasets/umls/')
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize 
import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import BertTokenizer, BertForMaskedLM
from nltk.tokenize import sent_tokenize
import nltk
import time
import pickle
from torch.utils.data import DataLoader
from torch.optim import AdamW
import random
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import hmean, gmean
import numpy as np

path = "/global/cscratch1/sd/ajaybati/model_ckptDS5.pickle"
print("done")


from concept_web import generate_lexicon

model = BertForMaskedLM.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def load_model(): #load trained model for inference
    model.load_state_dict(torch.load(path,map_location=torch.device('cpu'))['model_state_dict'])
    model.eval()
    return model

model = load_model()

def get_model_input(sents,n_percent_mask=0.0):
    input_ids_real = []
    att = []
    compare = []
    mask_indices = []
    for sent in sent_tokenize(sents):
        mask = []
        encoded_dict = tokenizer.encode_plus(
            sent,                      # Sentence to encode.
            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
            max_length = 128,
            truncation = True,# Pad & truncate all sentences.
            pad_to_max_length = True,
            return_attention_mask = True,   # Construct attn. masks.
            return_tensors = 'pt',     # Return pytorch tensors.
       )
        input_ids = encoded_dict['input_ids']
        compare.append(input_ids)
        attention_masks = encoded_dict['attention_mask']
        att.append(attention_masks)
        input_ids_part = []
        for step,word in enumerate(input_ids[0]):
            if int(word) != 101 and int(word) != 102:
                rando = random.random()
                random.seed()
                if rando < n_percent_mask and int(word)!=0:
                    mask.append(step)
                    input_ids_part.append(103)
                else:
                    input_ids_part.append(int(word))
            else:
                input_ids_part.append(int(word))
        input_ids_part = torch.tensor(input_ids_part).view(1,128)
        input_ids_real.append(input_ids_part)
        mask_indices.append(mask)

    if len(input_ids_real)>1:
        bert_input = torch.cat(tuple(input_ids_real),0)
        att = torch.cat(tuple(att),0)
        compare = torch.cat(tuple(compare),0)
    else:
        bert_input = input_ids_real[0]
        att = att[0]
        compare = compare[0]

    return bert_input, att, compare, mask_indices, tokenizer.tokenize(sent)

def bert_model_output(model, bert_input, att, compare, mask_indices):
    loss, predictions = model(bert_input,attention_mask = att, masked_lm_labels = compare)
    accuracy, bscore = calc_accuracy(predictions, compare, mask_indices)

    return {"loss":loss,
            "predictions": predictions,
            "performance":[accuracy, bscore]}


def filter_sent(example_sent):  
    stop_words = set(stopwords.words('english')) 
    punt = ["!",'#','$','&','(',')','*','+','-','.',':',';','<','=','>','?','@','[',']','^','_','`','{','|','}','~',',']
    example_sent = example_sent.lower()
    word_tokens = word_tokenize(example_sent) 

    filtered_sentence = [w for w in word_tokens if not w in stop_words and not w in punt] 

    sentence = ''
    for word in filtered_sentence:
        sentence+=word+' '
    return sentence


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output[0].detach()
    return hook
model.bert.encoder.register_forward_hook(get_activation("encoder")) 
from sklearn.metrics.pairwise import cosine_distances as cosine_distances, cosine_similarity

str1='she complained of nausea with occasional vomitting and has been unable to keep down any of her oral medications.'
str1 = filter_sent(str1)

model_input, att, compare, mask_indices, tokens1 = get_model_input(str1, n_percent_mask=0.0)
out = model(model_input,attention_mask = att, masked_lm_labels = compare)
out1 = activation["encoder"][0][0:len(tokens1)]
fat = []
count = []

def is_SUICIDE(x):
    count.append(0)
    
    start = time.time()
    try:
        str2 = x.lower()
        str2 = filter_sent(str2)
        model_input2, att2, compare2, mask_indices2, tokens2 = get_model_input(str2, n_percent_mask=0.0)
        out2 = model(model_input2,attention_mask = att2, masked_lm_labels = compare2)
        out3 = activation["encoder"][0][0:len(tokens2)]
        cos = cosine_distances(out1,out3)

        cos = cos.flatten()

        hm = hmean(cos)
    except:
        print('here')
        hm = 1.0


    # out1 = out1[0:length]
    # out3=out3[0:length]



    
    end = time.time()
    if len(count)%1000==0:
        print('done through '+str(len(count))+' out of 11 mil')
        print('One query takes '+str(end-start))
    boolio = True if hm<=0.65 else False
    if boolio:
        print('='*80)
        print(x)
        print('='*80)
    if hm<=0.55:
        print('*'*80+'winner!!!!!')
    return boolio


print("Query String: she complained of nausea with occasional vomitting and has been unable to keep down any of her oral medications.")
lexicon = generate_lexicon(is_SUICIDE, verbose=True)
print(lexicon)
print(type(lexicon))
lexicon.to_pickle('/global/homes/a/ajaybati/lexicon.pickle')

lexicon.to_parquet("/global/homes/a/ajaybati/lexicon.parquet", index=False) 

