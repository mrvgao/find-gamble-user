all_char = {}

with open('database/train_corpus.tsv', encoding='utf-8') as f:
    for ii, line in enumerate(f):
        if ii == 0: continue
        char = line.split('\t')[0]
        all_char[char] = True


vocab_size = len(all_char)
print('vocab_size: {}'.format(vocab_size))

with open('database/vocab_size.txt', 'w') as f:
    f.write('vocab_size:{}'.format(vocab_size))