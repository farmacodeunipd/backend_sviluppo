import csv

# Define the paths
csv_file_path = './db/dataset/anaart.csv'
output_csv_file_path = './db/dataset/anaart_new.csv'
image_folder = '/dataset/imgs/'  # Assuming all images are in this folder

# Open the original CSV file for reading and the modified CSV file for writing
with open(csv_file_path, 'r', newline='') as csv_file, open(output_csv_file_path, 'w', newline='') as output_csv_file:
    # Create CSV reader and writer objects
    csv_reader = csv.reader(csv_file)
    csv_writer = csv.writer(output_csv_file)

    # Read the header row
    header = next(csv_reader)
    header.append('img_path')  # Add img_path to the header
    csv_writer.writerow(header)  # Write the modified header to the output CSV file

    # Iterate over each row in the CSV file
    for row in csv_reader:
        # Extract the value from the 'cod_art' column
        cod_art = row[0]
        # Construct the image path based on the cod_art
        img_path = f"{image_folder}{cod_art}.jpg"
        # Append the image path to the current row
        row.append(img_path)
        # Write the modified row to the output CSV file
        csv_writer.writerow(row)

print("CSV file updated successfully!")