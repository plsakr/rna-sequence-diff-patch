import json

nucleotides = ['A', 'G', 'C', 'U', 'Y', 'R', 'W', 'S', 'K', 'M', 'D', 'V', 'H', 'B', 'N']

data = {'insert': 1.0, 'delete': 1.0, 'update': {}}

for f in nucleotides:
    data['update'][f] = {}

for f in nucleotides:
    for t in nucleotides:
        cost = float(input(f'From {f} To: {t}:'))
        data['update'][f][t] = cost

with open('costs.json', 'w') as f:
    json.dump(data, f)
