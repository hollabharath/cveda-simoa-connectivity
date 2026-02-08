import os
import glob
import numpy as np
import pandas as pd

# --- Configuration ---
# Directory containing the .netcc files
INPUT_DIR = "Yeo17/"
# File pattern to match
FILE_PATTERN = "sub-*.netcc"
# Name for the final output CSV file
OUTPUT_CSV = "master_connectivity_matrix.csv"
# The number of ROIs (networks)
N_ROIS = 17

def parse_fz_matrix(filepath):
    """
    Parses a .netcc file to extract the upper triangle of the FZ matrix.

    Args:
        filepath (str): The full path to the .netcc file.

    Returns:
        numpy.ndarray: A 1D array containing the flattened upper-triangle
                       values of the FZ matrix, or None if not found.
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Find the start of the FZ matrix
        try:
            fz_start_index = lines.index("# FZ\n") + 1
        except ValueError:
            print(f"Warning: '# FZ' marker not found in {filepath}. Skipping.")
            return None

        # Extract the 17 lines of the FZ matrix
        matrix_lines = lines[fz_start_index : fz_start_index + N_ROIS]
        
        if len(matrix_lines) < N_ROIS:
            print(f"Warning: Incomplete FZ matrix in {filepath}. Skipping.")
            return None

        # Convert the string lines into a NumPy array of floats
        matrix = np.loadtxt(matrix_lines)

        # Get the indices of the upper triangle, excluding the diagonal (k=1)
        row_indices, col_indices = np.triu_indices(N_ROIS, k=1)

        # Extract the values from the upper triangle
        upper_triangle_values = matrix[row_indices, col_indices]

        return upper_triangle_values

    except Exception as e:
        print(f"Error processing file {filepath}: {e}")
        return None

def create_header():
    """
    Creates the header row with ROI-pair labels (e.g., ROI1-ROI2).

    Returns:
        list: A list of strings for the CSV header.
    """
    header = []
    for i in range(1, N_ROIS + 1):
        for j in range(i + 1, N_ROIS + 1):
            header.append(f"ROI{i}-ROI{j}")
    return header

def main():
    """
    Main function to find files, process them, and create the master CSV.
    """
    print("Starting data extraction...")
    
    # Get a list of all .netcc files matching the pattern
    search_path = os.path.join(INPUT_DIR, FILE_PATTERN)
    file_list = sorted(glob.glob(search_path))

    if not file_list:
        print(f"Error: No files found matching the pattern '{search_path}'.")
        print("Please check the INPUT_DIR and FILE_PATTERN variables.")
        return

    print(f"Found {len(file_list)} files to process.")

    all_subjects_data = []
    subject_ids = []

    # Process each file
    for filepath in file_list:
        # Extract subject ID from the filename
        basename = os.path.basename(filepath)
        sub_id = basename.split('.')[0] # Assumes format is sub-ID.netcc

        # Parse the file to get the connectivity data
        fz_values = parse_fz_matrix(filepath)

        if fz_values is not None:
            all_subjects_data.append(fz_values)
            subject_ids.append(sub_id)

    if not all_subjects_data:
        print("No data was successfully extracted. Exiting.")
        return

    # Create the header for the CSV file
    header = create_header()

    # Create a pandas DataFrame
    df = pd.DataFrame(all_subjects_data, index=subject_ids, columns=header)
    df.index.name = "subject_id"

    # Save the DataFrame to a CSV file
    df.to_csv(OUTPUT_CSV)

    print("-" * 30)
    print(f"Processing complete!")
    print(f"Master connectivity matrix saved to: {OUTPUT_CSV}")
    print(f"Dimensions of the final matrix: {df.shape[0]} subjects, {df.shape[1]} ROI pairs.")

if __name__ == "__main__":
    main()
