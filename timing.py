from StringEditDistance import wagnerFisher, create_paths, generate_es, patching, generate_rev_es
import random
import time


def random_nucleotide(length: int):

    nucleotides = ['A', 'G', 'C', 'U', 'Y', 'R', 'W', 'S', 'K', 'M', 'D', 'V', 'H', 'B', 'N']

    return "".join(random.choices(nucleotides, k=length))

time_wagner_fischer = []
time_paths = []
time_es = []
time_patching = []
time_reversing = []



for i in range(10, 260, 10):
    seq1 = random_nucleotide(i)
    seq2 = random_nucleotide(i)
    print(f'--------------- SEQUENCES OF LENGTH {i} -----------')
    print('Seq1:', seq1)
    print('Seq2:', seq2)

    start_time = time.time()
    dp = wagnerFisher(seq1, seq2)
    end_time = time.time()
    time_wagner_fischer.append((end_time - start_time)*1000)
    print('WF Done')

    start_time = time.time()
    paths = create_paths(dp)
    end_time = time.time()
    time_paths.append((end_time - start_time)*1000)
    print('Create Paths Done')
    
    start_time = time.time()
    for p in paths:
        editScripts = generate_es(p, seq1, seq2)
    end_time = time.time() 
    time_es.append((end_time - start_time)*1000)
    print('Generate ES Done')
    
    start_time = time.time()
    patched = patching(editScripts, seq1)
    
    if patched != seq2:
        print('SOMETHING AWFUL HAPPENED!!!!!!')
        
    end_time = time.time()
    time_patching.append((end_time-start_time)*1000)
    print('Patching Done')

    start_time = time.time()
    reversed = generate_rev_es(editScripts)
    end_time = time.time()
    time_reversing.append((end_time-start_time)*1000)
    print('Reversing Done')
    

print(time_wagner_fischer)
print(time_paths)
print(time_es)
print(time_patching)
print(time_reversing)
