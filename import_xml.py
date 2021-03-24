import xml.etree.ElementTree as ET


def import_xml(file_name):
    tree = ET.parse(file_name)
    imported_sequences = {}
    root = tree.getroot()

    for entry in root.findall('entry'):
        rnaId = entry.get('id')
        sequence = entry.find('RNAseq').text
        sequence = sequence.replace('T', 'U')
        sequence = sequence.replace('X', 'N')
        imported_sequences[rnaId] = sequence


    return imported_sequences
