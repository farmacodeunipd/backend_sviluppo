# Questo file serve esclusivamente per trasformare i file .txt in file .csv

import pandas as pd
import os

# Funzione che itera tutti i file in una cartella e salva i loro nomi
def get_files_in_folder(folder_path):
    file_names = []
    for file in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file)):
            file_names.append(file)
    return file_names

# Funzione che rimuove l'ultimo carattere da ogni riga di un file
def remove_last_character(input_file, output_file):
    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    modified_lines = [line.rstrip('\n')[:-1] for line in lines]

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for line in modified_lines:
            outfile.write(line + '\n')

# Funzione che rimuove " e trasforma il file in formato .csv
def txt_to_csv(input_file, output_file):

    df = pd.read_csv(input_file, delimiter=';', header=None) # CAMBIARE DELIMITERE SE DIVERSO DA ;

    df = df.applymap(lambda x: str(x).replace('"', '')) # CAMBIARE CARATTERE/I DA RIMUOVERE O COMMENTARE

    df.to_csv(output_file, index=False, header=False)

file_names = get_files_in_folder('dataset') # CAMBIARE IL NOME DEL FOLDER SE I DATI SONO IN UN FOLDER NUOVO, ASSICURARSI CHE CI SIANO SOLO FILE.TXT NEL FOLDER

file_names = [file_name.rstrip('\n')[:-4] for file_name in file_names]

for file_name in file_names:

    input_file = 'dataset/' + file_name + '.txt'
    output_file = 'dataset/' + file_name + '.csv'
    remove_last_character(input_file, output_file)
    txt_to_csv(output_file, output_file)