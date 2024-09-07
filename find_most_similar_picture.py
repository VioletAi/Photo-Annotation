import os
import numpy as np

def read_matrix_from_file(file_path):
    """ Reads a matrix from a text file. """
    return np.loadtxt(file_path)

def find_most_similar_matrix(target_matrix, folder_path):
    """
    Finds the matrix most similar to the matrix in target_path among matrices in the folder_path.

    Args:
    target_path (str): Path to the target matrix file.
    folder_path (str): Path to the folder containing matrix files.

    Returns:
    str: File name of the most similar matrix.
    np.array: The most similar matrix.
    """
    # Read the target matrix
    # target_matrix = read_matrix_from_file(target_path)
    
    min_distance = float('inf')
    most_similar_matrix = None
    most_similar_file = ""

    # Iterate through all files in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        
        # Read the matrix from the current file
        matrix = read_matrix_from_file(file_path)
        
        # Calculate the Frobenius norm of the difference
        distance = np.linalg.norm(target_matrix - matrix)
        
        # Update the most similar matrix if the current one is closer
        if distance < min_distance:
            min_distance = distance
            most_similar_matrix = matrix
            most_similar_file = file_name
    
    return most_similar_file,most_similar_matrix


if __name__=="__main__":
    # Example usage
    folder_path = 'path_to_folder'  # Replace with the path to your folder
    target_path = 'path_to_target_matrix_file'  # Replace with the path to your target matrix file

    similar_file, similar_matrix = find_most_similar_matrix(target_path, folder_path)
    print(f"The most similar matrix is in file: {similar_file}")
    print("Matrix:")
    print(similar_matrix)
