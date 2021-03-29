# RNA Sequences Diff and Patching Tool

## Tool Introduction:
This tool can calculate the edit distance between two RNA sequences, generate the corresponding edit script and patch an
RNA sequence into another. 


## Usage / Execution:

_Note: The data file ocu.fa needs to be inside a folder called data in order for the app to run_

The project required the use of Python (tested with Python v3.8). The required libraries are listed in `requirements.txt`
To install them all at once using pip, execute the following command:

`pip install -r requirements.txt`

After the requirements are downloaded. You can run the tool by running the following command:

`python gui.py`


## Input Formats:

The tool allows the use of a SeqXML file as an input. However:
 * Each entry in the file should only contain an `<RNASeq>` tag (exactly one). `<DNASeq>` tags are not compatible.
 * The `id` attribute in each entry is required, and will be the only thing shown in the GUI to differentiate each sequence.

The tool also allows selecting a sequence from the given .fa file (ocu.fa), in addition to simply providing the required
sequence directly in the GUI.
