import os
import torch
import transformers
import pandas as pd 
from functools import partial
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from collate_functions import collate_to_max_length

class NoReCDataset(Dataset):
    def __init__(self, dataset_path: str = 'data/combined_data.csv', max_length: int = 512):
        df = pd.read_csv(dataset_path)
        self.texts = list(df.text)
        self.targets = list(df.label)
        self.tokenizer = BertTokenizer.from_pretrained('ltgoslo/norbert2')
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        sentence = str(self.texts[item])
        label = self.targets[item]
        input_ids = self.tokenizer.encode(sentence, add_special_tokens=False)
        if len(input_ids) > self.max_length - 2:
                input_ids = input_ids[:self.max_length - 2]
        length = torch.LongTensor([len(input_ids) + 2])
        input_ids = torch.LongTensor([0] + input_ids + [2])
        label = torch.LongTensor([int(label)])
        return input_ids, label, length
    



def unit_test():
    dataset = NoReCDataset()

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=10,
        num_workers=0,
        shuffle=False,
        collate_fn=partial(collate_to_max_length, fill_values=[1, 0, 0])
    )
    for input_ids, label, length, start_index, end_index, span_mask in dataloader:
        print('----------INPUT IDS----------')
        print(input_ids.shape)
        print('----------START IDX----------')
        print(start_index.shape)
        print('----------END IDX----------')
        print(end_index.shape)
        print('----------SPAN MASK----------')
        print(span_mask.shape)
        print('----------LABEL----------')
        print(label.view(-1).shape)
        print()


if __name__ == '__main__':
    unit_test()