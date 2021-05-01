import pickle
import time
from pymongo import MongoClient
import bson
from IRMethods import convert_to_tf_vector

#This script is used to parse through fa files in order to import their corresponding dataset
#Note that the .fa files for this project (demo included) were imported from piRNA db

#The processed dataset will be stored in a dictionary called data where keys are the IDs/titles of the processed RNA
#sequences and the values are the corresponding RNA Sequence for each ID/title
data = {}

mongoclient = MongoClient()
db = mongoclient.rna_db
collection = db.sequences
#Each sequence in the database has a unique identifier which will be stored as title
#and its corresponding sequence will be stored in the currentSequence
currentSequence = ''
currentTitle = ''

imported = 500
#In order to decrease delays and make the shown RNA sequences more visible, we limited ourselves to a maximum of 500
#RNA sequences. The variable imported is used as a down-counter.

#Time is used to check how much time does this processing take
start = time.time()

#locating the file in the directory
with open('./data/ocu.fa') as f:

    for line in f:
        #Usually, titles/ids of RNA Sequences have this format '>'
        if line[0] == '>':
            #If the current sequence is empty: no RNA sequence is currently being processed
            if currentSequence != '':
                #Select the sequence at the current id/title, decrement the counter
                #and reset the current Sequence (clear it)
                data[currentTitle] = currentSequence
                # collection.insert_one({'sequence': currentSequence, 'tf': bson.Binary(pickle.dumps(convert_to_tf_vector(currentSequence)))})
                imported -= 1
                currentSequence = ''

            if imported <= 0:
                break
            currentTitle = line[1:-1]
        else:
            #Select the currentSequence and replace the character T with U and X with N
            #(based on our research X is homomorphic to N and T is homomorphic to U)
            currentSequence = currentSequence + line[:-1]
            currentSequence = currentSequence.replace('T', 'U')
            currentSequence = currentSequence.replace('X', 'N')

#end is used to mark the time when we finished processing this file
end = time.time()

#Printing the time it took to finish processing
print(f'importing fa file took {end-start} s.')
#Printing the nbr of unique sequences that we got
print(f'imported {len(data.keys())} sequences')

#get_all_keys will return all ids/titles of RNA Sequences that are stored in our data dictionary
def get_all_keys():
    return list(data.keys())

#get_seq will fetch the RNA sequence from the dictionary at a given title/id specified by the user and return it
def get_seq(key):
    return data[key]
