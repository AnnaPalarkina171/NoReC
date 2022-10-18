from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--label")
parser.add_argument("--threshold")
parser.add_argument("--model")
args = parser.parse_args()

label = int(args.label)
threshold = float(args.threshold)
model = args.model

print(f'This is augmentation for label {str(label)} and threshold {str(threshold)}')

import pandas as pd 
from transformers import pipeline
from tqdm.auto import tqdm
import string
from random import choice
import nltk
from typing import List
import numpy as np 

import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv('combined_data.csv')

if model == 'roberta':
    unmasker = pipeline('fill-mask', model='xlm-roberta-large')
if model == 'norbert':
    unmasker = pipeline('fill-mask', model='ltgoslo/norbert2')


punctuation = [i for i in string.punctuation]
punctuation.append('...')
punctuation.append('``')
punctuation.append('\'\'')

with open('Fullform_Negative_lexicon.txt', 'r') as n:
    negatives = n.read().splitlines()

with open('Fullform_Positive_lexicon.txt', 'r') as p:
    positives = p.read().splitlines()

def matching_polarity(token, candidate_token):
    if (token in positives and candidate_token in positives)\
        or (token in negatives and candidate_token in negatives) \
        or ((token not in negatives and  candidate_token not in negatives) and (token not in positives and  candidate_token not in positives)):
        return True
    else:
        return False 

def check_fit(threshold, token, possible_replacement):
    if (token not in punctuation and possible_replacement['token_str'] not in punctuation)\
        and (possible_replacement['token_str'].lower() != token.lower())\
        and (possible_replacement['score'] >= threshold)\
        and (matching_polarity(token.lower(), possible_replacement['token_str'].lower()) == True):
        return True
    else:
        return False 

def augment_text(text: str, threshold:float):

    sentences = nltk.sent_tokenize(text, language='norwegian')
    sent_tokens = [nltk.word_tokenize(sentence, language='norwegian') for sentence in sentences]
    new_sent_tokens = []
    
    for sentence in sent_tokens:
        new_sentence=[]
        for i_t, token in enumerate(sentence):
            not_masked = True
            masked_tokens = sentence.copy()
            if model == 'roberta':
                masked_tokens[i_t] = '<mask>'
            if model == 'norbert':
                masked_tokens[i_t] = '[MASK]'
            unmasked = unmasker(' '.join(masked_tokens))
            for possible_replacement in unmasked:
                if check_fit(threshold, token, possible_replacement):
                    new_sentence.append(possible_replacement['token_str'])
                    #aug_tokens+=1
                    not_masked = False
                    break
            if not_masked:
                new_sentence.append(token)
                
        #print(aug_tokens/len(new_sentence))
        new_sent_tokens+=new_sentence
        

    return new_sent_tokens

df_to_augment = df[df.split=='train'][df.label == label]

for i, row in tqdm(df_to_augment.iterrows()):
    new_tokens = augment_text(text = row.text, threshold=threshold)
    for modulo in range(8):
        if len(new_tokens)//800 == modulo:
            splits = np.array_split(new_tokens, modulo+1)
            for tokens in splits:
                row.text = ' '.join(tokens)
                df = df.append(row)

df.to_csv(f'augmented_splited_{str(label)}_{str(threshold)[-1]}.csv', index=False)
print('DataFrame saved!')
