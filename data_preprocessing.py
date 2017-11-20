"""
Processing the data as (char, if_gamble) dataset.

And use these data to train a model which could predict the char if is a gamble user's id.

If we train the model fitful, we get the embedding for each char.

And we could define some high probability words and use the heuristic method getting the related words.

"""

import pandas as pd
from itertools import product


file = open('database/train_corpus.tsv', 'w', encoding='utf-8')


csv_data = pd.read_csv('database/gamble_id_table.csv', encoding='utf-8')

if_gamble = 'if_gamble'
old_nickname = 'old_nickname'
new_nickname = 'new_nickname'

user_id_new = csv_data[new_nickname].tolist()
user_id_old = csv_data[old_nickname].tolist()
user_if_gamble = csv_data[if_gamble].tolist()


def produce_train_data(word1, word2, gamble):
    if pd.isna(word1) or pd.isna(word2): return 0
    else:
        count = 0
        for c1, c2, g in product(word1, word2, [gamble]):
            file.write("{}\t{}\n".format(c1, g))
            file.write("{}\t{}\n".format(c2, g))
            count += 2
        return count


num = 0
for w1, w2, g in zip(user_id_new, user_id_old, user_if_gamble):
    count = produce_train_data(w1, w2, g)
    num += count
    print(num)


print('processing done!')
