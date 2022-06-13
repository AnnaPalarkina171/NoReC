from lib2to3.pgen2.pgen import DFAState
import pandas as pd 
from summa.summarizer import summarize
from tqdm.auto import tqdm
tqdm.pandas()
import matplotlib.pyplot as plt

data = pd.read_csv('combined_data.csv')
data['len_text_tokens'] = data['text'].apply(lambda text: len(str(text).split(' ')))

data['text'] = data['text'].progress_apply(lambda text: summarize(text, language='norwegian') if len(text.split(' '))>512 else text)

data.to_csv('summarized_data.csv', index=False)


# df = pd.read_csv('combined_data.csv')
# df['text'] = df['text'].progress_apply(lambda text: summarize(text, language='norwegian'))

# df.to_csv('summarized_data.csv', index=False)

# dff = pd.read_csv('combined_data.csv')
# lenss = dff['text'].apply(lambda text: len(str(text).split(' ')))
# print('min: ', lenss.min())
# print('average: ', lenss.mean())
# print('max: ', lenss.max())

# print('\n\n')


# df = pd.read_csv('summarized_data.csv')
# lens = df['text'].apply(lambda text: len(str(text).split(' ')))
# print('min sum: ', lens.min())
# print('average sum: ', lens.mean())
# print('max sum: ', lens.max())


# min:  3
# average:  404.0111202824781
# max:  3943


# min sum:  1
# average sum:  128.43566286054937
# max sum:  1451

