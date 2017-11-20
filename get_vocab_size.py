all_char = {}

with open('database/train_corpus.txt', encoding='utf-8') as f:
    for line in f:
        char = line.split('\t')[0]
        all_char[char] = True


print('vocab_size: {}'.format(len(all_char)))

with open('database/vocab_size.txt', 'w') as f:
    f.write('vocab_size:{}'.format(len(all_char)))