"""
Processing the data as (char, if_gamble) dataset.

And use these data to train a model which could predict the char if is a gamble user's id.

If we train the model fitful, we get the embedding for each char.

And we could define some high probability words and use the heuristic method getting the related words.

"""

import pandas as pd
from itertools import product
from collections import Counter


file = open('database/train_corpus.tsv', 'w', encoding='utf-8')


csv_data = pd.read_csv('database/gamble_id_table.csv', encoding='utf-8')

if_gamble = 'if_gamble'
old_nickname = 'old_nickname'
new_nickname = 'new_nickname'

user_id_new = csv_data[new_nickname].tolist()
user_id_old = csv_data[old_nickname].tolist()
user_if_gamble = csv_data[if_gamble].tolist()

num = 0

gamble_time_counter = Counter([])
not_gamble_chars = []

for w1, w2, g in zip(user_id_new, user_id_old, user_if_gamble):
    if pd.isna(w1): w1 = ""
    if pd.isna(w2): w2 = ""

    w = w1 + w2
    if w == "": continue

    print('{}: {}'.format(g, w))
    if str(g) == '1':
        gamble_time_counter += Counter(w)
    if str(g) == '0':
        not_gamble_chars += list(w)

total_count = sum(gamble_time_counter.values())

gamble_time_map = dict(gamble_time_counter)
gamble_time_map = {k: v/total_count for k, v in gamble_time_map.items()}

for c in not_gamble_chars:
    if c not in gamble_time_counter:
        gamble_time_map[c] = 0

for k, v in gamble_time_map.items():
    file.write('{}\t{}\n'.format(k, v))

print('processing done!')
