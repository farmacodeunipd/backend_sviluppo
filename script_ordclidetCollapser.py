import pandas as pd
import numpy as np

def process_file(input_file, output_file):
    # Read the input file into a pandas DataFrame, replacing commas with dots
    df = pd.read_csv(input_file, delimiter=';', thousands=',', decimal='.')

    # Group by 'cod_cli' and 'cod_art' and sum the 'qta_ordinata' values
    grouped_df = df.groupby(['cod_cli', 'cod_art'], as_index=False)['qta_ordinata'].sum()
    
    # Multiply the original quantity by the logarithm
    grouped_df['qta_ordinata'] = np.log10(grouped_df['qta_ordinata'])
    
    # Cap the final value at 2
    grouped_df['qta_ordinata'] = np.minimum(grouped_df['qta_ordinata'], 2)
    
    # Set minimum value at 0.0001
    grouped_df['qta_ordinata'] = np.maximum(grouped_df['qta_ordinata'], 0.0001)
    
    # Round the values in 'qta_ordinata' to 4 decimal places
    grouped_df['qta_ordinata'] = grouped_df['qta_ordinata'].round(4)

    # Convert 'qta_ordinata' column to string
    grouped_df['qta_ordinata'] = grouped_df['qta_ordinata'].astype(str)

    # Save the result to the output file
    grouped_df.to_csv(output_file, index=False)

# Replace '.\dataset\dataset\ordclidet.csv' and 'output_file.csv' with your actual file names
process_file('.\dataset\dataset\ordclidet.csv', 'output_file.csv')