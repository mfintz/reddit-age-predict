from nltk.tokenize import RegexpTokenizer
import codecs
import os
import pandas as pd
import pickle
from datetime import datetime


# Collect sentences from mergedFiles into a single DataFrame
# Columns: age, author, sentence, sent_length
def collect_data(input_path):
    dir_files = os.listdir(input_path)
    # if not os.path.exists(input_path + "final\\"):
    #     os.makedirs(input_path + "final\\")

    tokenizer = RegexpTokenizer(r'\w+')
    df = pd.DataFrame()
    for f in dir_files:
        age = f[:f.find('.')]
        if f.endswith('.merged'):
            # source file
            src_file = codecs.open('{}{}'.format(input_path, f), 'r', 'utf-8')

            authors, sentences, sents_len = [], [], []
            for line in src_file:
                sentence = line[line.find('] ') + 2:]
                authors.append(line[line.find('[')+1:line.find('] ')])
                sentences.append(sentence)
                sents_len.append(len(tokenizer.tokenize(sentence)))

            df = df.append(pd.DataFrame({'author': authors, 'age': [age]*len(sentences), 'sent_length': sents_len, 'sentence': sentences}))
            print('Loaded {} data...'.format(f))
    return df.sample(frac=1)


# This function creates a new dataframe, columns = age , batch(author, sent_length, sentence)
# batch is a list of sentences with total number of words 'equals' to 'batch_size'
# sentences are kept whole.
def create_batches(df, batch_size=500, dwn_smpl=True):

    num_words = df.groupby(['age'])['sent_length'].sum()
    num_batchs = [n//batch_size for n in num_words]
    print(num_batchs)
    if dwn_smpl:
        min_batchs = min(num_batchs)
        cat_sizes = zip(num_words.index, num_words, [min_batchs]*len(num_batchs))
    else:
        cat_sizes = zip(num_words.index, num_words, num_batchs)
    batch_df = pd.DataFrame(columns=['age', 'batch'])

    for s in cat_sizes:
        cat_df = df[(df['age'] == s[0]) & (df['sent_length']!=0)]
        cat_df = cat_df.reset_index()
        cat_df['cumsum'] = cat_df['sent_length'].cumsum()
        i, j = 0, 1
        batch = []

        while (dwn_smpl and i < len(cat_df) and j <= s[2]) or (not dwn_smpl and i < len(cat_df)):
            batch.append((cat_df['author'][i], cat_df['sent_length'][i], cat_df['sentence'][i]))  # maybe concat is better, but essential for the avg sentence length feature
            if cat_df['cumsum'][i] > batch_size*j:
                batch_df.loc[len(batch_df)] = {'age': s[0],
                                               'batch': batch}  # pd.DataFrame({'age': s[0], 'batch': batch})
                print('age {} batch {}'.format(s[0], j))
                batch = []
                j += 1
            i += 1
        if not len(batch):       # Add last batch
            batch_df.loc[len(batch_df)] = {'age': s[0], 'batch': batch}
    batch_df = batch_df.sample(frac=1)
    return batch_df

# This function returns a balanced dataframe - each age category of the same length
# This function receives the original dataframe (Columns: age, author, sentence, sent_length)
def down_sample(df):
    min_sample =  df.groupby(['age']).size().min()
    sampled_df = pd.DataFrame()
    for age in df['age'].unique():
        sampled_df = sampled_df.append(df[df['age'] == age].sample(n=min_sample))
    return sampled_df

def handle_batch(batch):
    avg_sent = 0
    sum_len = 0
    text = ''
    for t in batch:
        sum_len += t[1]
        text += ' '+t[2]
    if len(batch):
        avg_sent = sum_len/len(batch)
    return (avg_sent, text)


def from_batch_to_dataset(batch_df):
    dataset = batch_df
    dataset['text'] = batch_df['batch'].apply(lambda b: handle_batch(b)[1])
    dataset['avg_sent'] = batch_df['batch'].apply(lambda b: handle_batch(b)[0])
    dataset.drop(columns=['batch'], inplace=True)
    return dataset



################ Collect data (done once) ####################################################
# print('Start data collection: '+datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
# data = collect_data('jsons\\texts\\splitsentsNLTKstyle\\cleanSents\\mergedFiles\\')
# print('Finish data collection: '+datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
# pickle.dump(data, open('data.p', 'wb'))
##############################################################################################

data = pickle.load(open('data.p', "rb"))
print(data.head())

# batch_df = create_batches(data, 2000)
# pickle.dump(batch_df, open('batch_df.p', 'wb'))
batch_df = pickle.load(open('batch_df.p', "rb"))
print(batch_df.head())

dataset = from_batch_to_dataset(batch_df)
print(dataset.head())

print(dataset.values)


# sampled_df = down_sample(dataset)



