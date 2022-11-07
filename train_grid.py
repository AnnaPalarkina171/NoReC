import transformers
from transformers import BertModel, BertTokenizer, AdamW, LongformerModel, BertForSequenceClassification, get_linear_schedule_with_warmup, AutoTokenizer, LongformerTokenizer
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from collections import defaultdict
from textwrap import wrap
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import torch.nn.functional as F
import warnings
import os
import random
from argparse import ArgumentParser


warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = ArgumentParser()
parser.add_argument("--lr")
parser.add_argument("--dropout")
parser.add_argument("--warmup")
args = parser.parse_args()

LR = float(args.lr) 
DROPOUT = float(args.dropout)
WARMUP = int(args.warmup)


print('THIS IS SIMPLE NORBERT MODEL using combined_data.csv')
print(f'LR = {LR}, dropout= {DROPOUT}, num_warmup_steps= {WARMUP}')

def seed_everything(seed_value=42):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
seed_everything()

PRE_TRAINED_MODEL_NAME = 'ltgoslo/norbert2'
MAX_LEN = 512   
BATCH_SIZE = 4
EPOCHS = 20


df = pd.read_csv('combined_data.csv')

class_names = df.label.unique()
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

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
      truncation=True,
      return_tensors='pt',
    )

    return {
      'text': text,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }

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

class SentimentClassifier(nn.Module):

  def __init__(self, n_classes):
    super(SentimentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    self.drop = nn.Dropout(p=DROPOUT)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

  def forward(self, input_ids, attention_mask):
    
    bert_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask,
      return_dict=False
    )
    last_hidden_state, pooled_output = bert_output
    output = self.drop(pooled_output)
    return self.out(output)


df_train = df[df['split'] == 'train']
df_val = df[df['split'] == 'dev']
df_test = df[df['split'] == 'test']

print(f'Train samples: {len(df_train)}')
print(f'Validation samples: {len(df_val)}')
print(f'Test samples: {len(df_test)}')

train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

model = SentimentClassifier(len(class_names))
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=LR, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=WARMUP,
  num_training_steps=total_steps
)
loss_fn = nn.CrossEntropyLoss().to(device)

def train_epoch(
  model,
  data_loader,
  loss_fn,
  optimizer,
  device,
  scheduler,
  n_examples
):

  y_true, y_pred = [], []
  model = model.train()
  losses = []
  correct_predictions = 0
  
  for d in data_loader:
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    targets = d["targets"].to(device)
    y_true += targets.tolist()
    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    _, preds = torch.max(outputs, dim=1)
    y_pred += preds.tolist()
    loss = loss_fn(outputs, targets)
    correct_predictions += torch.sum(preds == targets)
    losses.append(loss.item())
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
  f1 = f1_score(y_true, y_pred, average='macro')

  return correct_predictions.double() / n_examples, np.mean(losses), f1

def eval_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()
  losses = []
  correct_predictions = 0
  y_true, y_pred = [], []
  with torch.no_grad():
    for d in data_loader:
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)
      y_true += targets.tolist()
      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)
      y_pred += preds.tolist()
      loss = loss_fn(outputs, targets)
      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())
  f1 = f1_score(y_true, y_pred, average='macro')
  report = classification_report(y_true, y_pred)
  return correct_predictions.double() / n_examples, np.mean(losses), f1, report
  
for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)
    train_acc, train_loss, train_f1 = train_epoch(
        model,
        train_data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        len(df_train)
    )
    print()
    print(f'Train loss {train_loss} accuracy {train_acc} f1 {train_f1}')

    val_acc, val_loss, val_f1, report = eval_model(
        model,
        val_data_loader,
        loss_fn,
        device,
        len(df_val)
    )
    print()
    print(f'Val   loss {val_loss} accuracy {val_acc} f1 {val_f1}')
    print(report)


def test_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()
  losses = []
  correct_predictions = 0
  y_true, y_pred = [], []
  with torch.no_grad():
    for d in tqdm(data_loader):
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)
      y_true += targets.tolist()
      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)
      y_pred += preds.tolist()
      loss = loss_fn(outputs, targets)
      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())
  f1 = f1_score(y_true, y_pred, average='macro')
  report = classification_report(y_true, y_pred)
  return correct_predictions.double() / n_examples, f1, report, y_true, y_pred



test_acc, test_f1, test_report, y_true, y_pred = test_model(
    model,
    test_data_loader,
    loss_fn,
    device,
    len(df_val)
  )

print()
print('-------------TESTINGS-----------------')
print()
print(f'Test accuracy {test_acc} f1 {test_f1}')
print(test_report)

print()
print()
print('Y TRUE: ', y_true)
print('Y PREDICTED', y_pred)
