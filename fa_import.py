import time

data = {}

currentSequence = ''
currentTitle = ''

imported = 100000
start = time.time()
with open('./data/ocu.fa') as f:
    for line in f:
        if line[0] == '>':

            if currentSequence != '':
                data[currentTitle] = currentSequence
                imported -= 1
                currentSequence = ''

            if imported <= 0:
                break
            currentTitle = line[1:-1]
        else:
            currentSequence = currentSequence + line[:-1]
            currentSequence = currentSequence.replace('T', 'U')
            currentSequence = currentSequence.replace('X', 'N')
end = time.time()

print(f'importing fa file took {end-start} s.')
print(f'imported {len(data.keys())} sequences')

def get_all_keys():
    return list(data.keys())


def get_seq(key):
    return data[key]
