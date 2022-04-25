import transformers
from transformers import BertModel, BertTokenizer, AdamW, LongformerModel, get_linear_schedule_with_warmup, AutoTokenizer, LongformerTokenizer
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PRE_TRAINED_MODEL_NAME = 'ltgoslo/norbert2'
MAX_LEN = 512   # longest text = 9111
RANDOM_SEED = 42
BATCH_SIZE = 32
EPOCH_N = 3

df = pd.read_csv('combined_data.csv')
df = df.head(100)
class_names = df.label.unique()
# tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
# long_model = LongformerModel.from_pretrained('allenai/longformer-base-4096')

print('Initializing model and tokenizer - BERT')
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)

class Dataset(Dataset):
  def __init__(self, texts, targets, tokenizer, max_len):
    self.texts = texts
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.texts)

  def __getitem__(self, item):
    text = str(self.texts[item])
    target = self.targets[item]
    encoding = self.tokenizer.encode_plus(
      text,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
    )
    return {
      'text': text,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }

df_train, df_val = train_test_split(
  df,
  test_size=0.2,
  random_state=RANDOM_SEED
)


def create_data_loader(df, tokenizer, max_len, batch_size):
  ds = Dataset(
    texts=df.text.to_numpy(),
    targets=df.label.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )
  return DataLoader(
    ds,
    batch_size=batch_size
  )


print('Creating train and validation dataloaders')
train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)

data = next(iter(train_data_loader))


class SentimentClassifier(nn.Module):

  def __init__(self, n_classes):
    super(SentimentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

  def forward(self, input_ids, attention_mask):
    
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    output = self.drop(pooled_output)
    return self.out(output)


print('Initializing custon class')
model = SentimentClassifier(len(class_names))
model = model.to(device)


print('Make input_ids and attention_mask')
input_ids = data['input_ids'].to(device)   # batch size x seq length
attention_mask = data['attention_mask'].to(device)   # batch size x seq length

print(F.softmax(model(input_ids, attention_mask), dim=1))











#PRE_TRAINED_MODEL_NAME = 'bert-base-cased'

# sample_txt = """
# Penny Dreadful har fått ufortjent lite oppmerksomhet her til lands 
# Serien bringer kjente figurer fra klassiske gotiske romaner sammen i én historie
# En begeistring for førtifem kompakte minutter med sånn passe avansert whodunit med konklusjon i hver episode har ført til at jeg har latt meg 
# """

# tokens = tokenizer.tokenize(sample_txt)
# token_ids = tokenizer.convert_tokens_to_ids(tokens)
# print(f' Sentence: {sample_txt}')
# print(f'   Tokens: {tokens}')
# print(f'Token IDs: {token_ids}')

# tokenizer.sep_token, tokenizer.sep_token_id
# tokenizer.cls_token, tokenizer.cls_token_id
# tokenizer.pad_token, tokenizer.pad_token_id
# tokenizer.unk_token, tokenizer.unk_token_id

# encoding = tokenizer.encode_plus(
#   sample_txt,
#   max_length=64,
#   add_special_tokens=True, # Add '[CLS]' and '[SEP]'
#   return_token_type_ids=False,
#   pad_to_max_length=True,
#   return_attention_mask=True,
#   return_tensors='pt',  # Return PyTorch tensors
# )

# print(encoding['attention_mask'].shape, encoding['attention_mask'],'\n\n')
# print(encoding['attention_mask'].flatten().shape, encoding['attention_mask'].flatten())

# print(encoding.keys())
# print(encoding['input_ids'])
# print(encoding['attention_mask'])
# tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])


