import pandas as pd
import numpy as np

# Questa funzione unisce acquisti e feedback, li collassa e trasforma la quantità acquistata in rating usando un logaritmo, questo script è sufficiente per preparare i dati per algoritmi matriciali
# usa le tabelle ORDCLIDET e ORDCLIDET_FEEDBACK

def process_file(input_file_og, input_file_fb, output_file):

    df1 = pd.read_csv(input_file_og, delimiter=',', thousands=',', decimal='.')
    df2 = pd.read_csv(input_file_fb, delimiter=',', thousands=',', decimal='.')
    df = pd.concat([df1, df2], ignore_index=True)

    # Group by 'cod_cli' and 'cod_art' and sum the 'qta_ordinata' values
    grouped_df = df.groupby(['cod_cli', 'cod_art'], as_index=False)['qta_ordinata'].sum()
    
    # Multiply the original quantity by the logarithm
    grouped_df['qta_ordinata'] = np.log10(grouped_df['qta_ordinata'])
    
    # Cap the final value at 2 & et minimum value at 0.0001
    grouped_df['qta_ordinata'] = np.minimum(grouped_df['qta_ordinata'], 2)
    grouped_df['qta_ordinata'] = np.maximum(grouped_df['qta_ordinata'], 0.0001)
    
    # Round the values in 'qta_ordinata' to 4 decimal places & convert 'qta_ordinata' column to string
    grouped_df['qta_ordinata'] = grouped_df['qta_ordinata'].round(4)
    grouped_df['qta_ordinata'] = grouped_df['qta_ordinata'].astype(str)

    # Rename the column from 'qta_ordinata' to 'rating'
    grouped_df = grouped_df.rename(columns={'qta_ordinata': 'rating'})

    # Save the result to the output file
    grouped_df.to_csv(output_file, index=False)

# Questa funzione aggiunge al file sufficiente agli algoritmi matriciali, anche le informazioni singole di ogni prodotto, rendendolo utilizzabile anche da NN
# usa il file data_preprocessed_matrix.csv e la tabella ANAART

def merge_files(file1, file2, output_file):
    # Read the first file into a DataFrame
    df1 = pd.read_csv(file1)

    # Read the second file into a DataFrame
    df2 = pd.read_csv(file2)

    # Ensure 'cod_art' column has the same data type in both DataFrames
    df1['cod_art'] = df1['cod_art'].astype(str)
    df2['cod_art'] = df2['cod_art'].astype(str)

    # Merge the DataFrames using an inner join on 'cod_art'
    merged_df = pd.merge(df1, df2, how='inner', on='cod_art')

    # Remove the useless columns
    merged_df.drop('des_art', axis=1, inplace=True)
    merged_df.drop('cod_sott_comm', axis=1, inplace=True)

    # Save the merged DataFrame to a new file
    merged_df.to_csv(output_file, index=False)

process_file('./dataset/ordclidet.csv', './dataset/ordclidet_feedback.csv', './NN/data_preprocessed_matrix.csv')

merge_files('./NN/data_preprocessed_matrix.csv', './dataset/anaart.csv', './NN/data_preprocessed_NN.csv')