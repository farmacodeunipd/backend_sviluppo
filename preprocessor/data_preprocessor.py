from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import mysql.connector
import csv
import os

class Preprocessor(ABC):
    tables_to_export = []
    
    @abstractmethod
    def retrieve_file(self, cursor, table_name, csv_path):
        pass
    
    def retrieve_file(self, cursor, table_name, csv_path):
        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()
        if rows:
            csv_file = os.path.join(csv_path, f"{table_name}.csv")
            with open(csv_file, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                # Write header
                csv_writer.writerow([description[0] for description in cursor.description])
                # Write rows
                csv_writer.writerows(rows)
            print(f"Table '{table_name}' exported to '{csv_file}'")
        else:
            print(f"No data found in table '{table_name}'")

class SVD_Preprocessor(Preprocessor):
    tables_to_export = ['ordclidet_feedback', 'ordclidet']
    
    def process_file(self, input_file_og, input_file_fb, output_file):
        df1 = pd.read_csv(input_file_og, delimiter=',', thousands=',', decimal='.')
        df2 = pd.read_csv(input_file_fb, delimiter=',', thousands=',', decimal='.')
        
        if not df2.empty:
            df = pd.concat([df1, df2], ignore_index=True)
        else:
            df = df1.copy()

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

class NN_Preprocessor(Preprocessor):
    tables_to_export = ['anaart', 'ordclidet_feedback', 'ordclidet', 'anacli']
    
    def process_file(self, input_file_og, input_file_fb, output_file):
        df1 = pd.read_csv(input_file_og, delimiter=',', thousands=',', decimal='.')
        df2 = pd.read_csv(input_file_fb, delimiter=',', thousands=',', decimal='.')
        
        if not df2.empty:
            df = pd.concat([df1, df2], ignore_index=True)
        else:
            df = df1.copy()

        grouped_df = df.groupby(['cod_cli', 'cod_art'], as_index=False)['qta_ordinata'].sum()
        grouped_df['qta_ordinata'] = np.log10(grouped_df['qta_ordinata'])
        grouped_df['qta_ordinata'] = np.minimum(grouped_df['qta_ordinata'], 2)
        grouped_df['qta_ordinata'] = np.maximum(grouped_df['qta_ordinata'], 0.0001)
        grouped_df['qta_ordinata'] = grouped_df['qta_ordinata'].round(4)
        grouped_df['qta_ordinata'] = grouped_df['qta_ordinata'].astype(str)
        grouped_df = grouped_df.rename(columns={'qta_ordinata': 'rating'})

        # Merge with additional information
        df_additional = pd.read_csv('./preprocessor/exported_csv/anaart.csv')
        merged_df = pd.merge(grouped_df, df_additional, how='inner', on='cod_art')
        merged_df.drop('des_art', axis=1, inplace=True)
        merged_df.drop('cod_sott_comm', axis=1, inplace=True)
        merged_df.sort_values(by=['cod_cli'])
        merged_df.to_csv(output_file, index=False)

class PreprocessorContext:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
    
    def set_preprocessor(self, preprocessor):
        self.preprocessor = preprocessor

    def process_file(self, input_file_og, input_file_fb, output_file):
        self.preprocessor.process_file(input_file_og, input_file_fb, output_file)

# Example usage
if __name__ == "__main__":
    svd_preprocessor = SVD_Preprocessor()
    nn_preprocessor = NN_Preprocessor()
    context = PreprocessorContext(nn_preprocessor)  # Initial strategy
    
    with mysql.connector.connect(
        host='localhost',  # <--- Enter your database details here
        user='myuser',
        password='mypassword',
        database='mydatabase',
        port='3306'
    ) as conn:
        cursor = conn.cursor()
        csv_folder = 'preprocessor/exported_csv'
        if not os.path.exists(csv_folder):
            os.makedirs(csv_folder)
            
        for table_name in context.preprocessor.tables_to_export:
            context.preprocessor.retrieve_file(cursor, table_name, csv_folder)
            
    context.process_file('./preprocessor/exported_csv/ordclidet.csv', './preprocessor/exported_csv/ordclidet_feedback.csv', './algoritmi/data_preprocessed_NN.csv')
    print(f"processed data ready")
    # Switch strategy if needed
    #     context.set_preprocessor(svd_preprocessor)

    #     for table_name in tables_to_export:
    #         context.preprocessor.retrieve_file(cursor, table_name, csv_folder)

    # # Process files using the selected strategy
    # context.process_file('.preprocessor/exported_csv/ordclidet.csv', '.preprocessor/exported_csv/ordclidet_feedback.csv', './NN/data_preprocessed_matrix.csv')
