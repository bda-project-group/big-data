# This is the code for the LSH project of TDT4305

import configparser  # for reading the parameters file
import sys  # for system errors and printouts
from pathlib import Path  # for paths of files
import os  # for reading the input data
import time  # for timing
import numpy as np
import numpy.typing as npt
from typing import TypedDict, Literal


class Parameters(TypedDict):
    data: Literal["test", "bbc"] | str
    k: int
    b: int
    r: int
    t: float
    naive: bool
    permutations: int


# Global parameters
parameter_file = "default_parameters.ini"  # the main parameters file
data_main_directory = Path("data")  # the main path were all the data directories are
parameters_dictionary: Parameters = {
    "data": "bbc",
    "k": 5,
    "b": 4,
    "r": 2,
    "t": 0.5,
    "naive": False,
    "permutations": 100,
}  # dictionary that holds the input parameters, key = parameter name, value = value
document_list: dict[
    int, str
] = dict()  # dictionary of the input documents, key = document id, value = the document


# DO NOT CHANGE THIS METHOD
# Reads the parameters of the project from the parameter file 'file'
# and stores them to the parameter dictionary 'parameters_dictionary'
def read_parameters():
    config = configparser.ConfigParser()
    config.read(parameter_file)
    for section in config.sections():
        for key in config[section]:
            if key == "data":
                parameters_dictionary[key] = config[section][key]
            elif key == "naive":
                parameters_dictionary[key] = bool(config[section][key])
            elif key == "t":
                parameters_dictionary[key] = float(config[section][key])
            else:
                parameters_dictionary[key] = int(config[section][key])


# DO NOT CHANGE THIS METHOD
# Reads all the documents in the 'data_path' and stores them in the dictionary 'document_list'
def read_data(data_path):
    for root, dirs, file in os.walk(data_path):
        for f in file:
            file_path = data_path / f
            doc = open(file_path).read().strip().replace("\n", " ")
            file_id = int(file_path.stem)
            document_list[file_id] = doc


# DO NOT CHANGE THIS METHOD
# Calculates the Jaccard Similarity between two documents represented as sets
def jaccard(doc1, doc2):
    return len(doc1.intersection(doc2)) / float(len(doc1.union(doc2)))


# DO NOT CHANGE THIS METHOD
# Define a function to map a 2D matrix coordinate into a 1D index.
def get_triangle_index(i, j, length):
    if i == j:  # that's an error.
        sys.stderr.write("Can't access triangle matrix with i == j")
        sys.exit(1)
    if j < i:  # just swap the values.
        temp = i
        i = j
        j = temp

    # Calculate the index within the triangular array. Taken from pg. 211 of:
    # http://infolab.stanford.edu/~ullman/mmds/ch6.pdf
    # adapted for a 0-based index.
    k = int(i * (length - (i + 1) / 2.0) + j - i) - 1

    return k


# DO NOT CHANGE THIS METHOD
# Calculates the similarities of all the combinations of documents and returns the similarity triangular matrix
def naive():
    docs_Sets = []  # holds the set of words of each document

    for doc in document_list.values():
        docs_Sets.append(set(doc.split()))

    # Using triangular array to store the similarities, avoiding half size and similarities of i==j
    num_elems = int(len(docs_Sets) * (len(docs_Sets) - 1) / 2)
    similarity_matrix = [0 for x in range(num_elems)]
    for i in range(len(docs_Sets)):
        for j in range(i + 1, len(docs_Sets)):
            similarity_matrix[get_triangle_index(i, j, len(docs_Sets))] = jaccard(
                docs_Sets[i], docs_Sets[j]
            )

    return similarity_matrix


K_shingles = list[set[str]]


# METHOD FOR TASK 1
# Creates the k-Shingles of each document and returns a list of them
def k_shingles() -> K_shingles:
    """
    Given an input array of length M, where each element, i, is a document of length N_i
    ```
    [
        "abcde", # document 1
        "fghij", # document 2
        "klmno", # document 3
    ]
    ```
    returns a list of length M, where each element, i, is a set of shingles of length k

    For example, with k=3
    ```
    [
        {"abc", "bcd", "cde"}, # document 1
        {"fgh", "ghi", "hij"}, # document 2
        {"klm", "lmn", "mno"}, # document 3
    ]
    """
    docs_k_shingles: K_shingles = []  # holds the k-shingles of each document

    # implement your code here
    for doc in document_list.values():
        k = parameters_dictionary["k"]
        shingles: set[str] = set()
        for i in range(len(doc) - k + 1):
            shingles.add(doc[i : i + k])

        docs_k_shingles.append(shingles)
    return docs_k_shingles


# METHOD FOR TASK 2
# Creates a signatures set of the documents from the k-shingles list
def signature_set(k_shingles: K_shingles) -> npt.NDArray[np.int64]:
    """
    Given an input array of length M, where each element, i, is a set of shingles of length N_i
    ```
    [
        [shingle1, shingle2, shingle3], # document1
        [shingle1, shingle2, shingle3], # document2
        [shingle3, shingle4, shingle5], # document3
    ]
    ```

    returns a matrix of size M x N, where N is the total number of unique shingles in all documents
    where each row j is a signature of a given shingle, and each column i is a document. The value at i, j is 1
    if and only if the shingle appears in document i.
    ```
    [
        [1, 1, 0], # shingle1
        [1, 1, 0], # shingle2
        [1, 1, 1], # shingle3
        [0, 0, 1], # shingle4
        [0, 0, 1], # shingle5
        # [document1, document2, document3]
    ]
    ```
    """
    unique_shingles: set[str] = set()
    for shingles in k_shingles:
        unique_shingles.update(shingles)

    signature_set: npt.NDArray[np.int64] = np.ndarray(
        [len(unique_shingles), len(k_shingles)], dtype=np.int64
    )

    for i, shingle in enumerate(unique_shingles):
        for j, document in enumerate(k_shingles):
            signature_set[i, j] = 1 if shingle in document else 0
    return signature_set


# METHOD FOR TASK 3
# Creates the minHash signatures after simulation of permutations
def min_hash(docs_signature_sets: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    number_of_permutations = parameters_dictionary["permutations"]

    min_hash_signatures: npt.NDArray[np.int64] = np.empty(
        shape=(number_of_permutations, docs_signature_sets.shape[1]), dtype=np.int64
    )

    rng = np.random.default_rng(seed=42)
    for i in range(number_of_permutations):
        print(f"permutation {i}")
        permutation = rng.permutation(docs_signature_sets)
        signature: npt.NDArray[np.int64] = np.argmax(permutation, axis=0)
        min_hash_signatures[i, :] = signature

    return min_hash_signatures


# METHOD FOR TASK 4
# Hashes the MinHash Signature Matrix into buckets and find candidate similar documents
def lsh(m_matrix: npt.NDArray[np.int64]):
    candidates = []  # list of candidate sets of documents for checking similarity

    # implement your code here

    return candidates


# METHOD FOR TASK 5
# Calculates the similarities of the candidate documents
def candidates_similarities(candidate_docs, min_hash_matrix: npt.NDArray[np.int64]):
    similarity_matrix = []

    # implement your code here

    return similarity_matrix


# METHOD FOR TASK 6
# Returns the document pairs of over t% similarity
def return_results(lsh_similarity_matrix):
    document_pairs = []

    # implement your code here

    return document_pairs


# METHOD FOR TASK 6
def count_false_neg_and_pos(lsh_similarity_matrix, naive_similarity_matrix):
    false_negatives = 0
    false_positives = 0

    # implement your code here

    return false_negatives, false_positives


# DO NOT CHANGE THIS METHOD
# The main method where all code starts
if __name__ == "__main__":
    # Reading the parameters
    read_parameters()

    # Reading the data
    print("Data reading...")
    data_folder = data_main_directory / parameters_dictionary["data"]
    t0 = time.time()
    read_data(data_folder)
    document_list = {k: document_list[k] for k in sorted(document_list)}
    t1 = time.time()
    print(len(document_list), "documents were read in", t1 - t0, "sec\n")

    # Naive
    naive_similarity_matrix = []
    if parameters_dictionary["naive"]:
        print("Starting to calculate the similarities of documents...")
        t2 = time.time()
        naive_similarity_matrix = naive()
        t3 = time.time()
        print(
            "Calculating the similarities of",
            len(naive_similarity_matrix),
            "combinations of documents took",
            t3 - t2,
            "sec\n",
        )

    # k-Shingles
    print("Starting to create all k-shingles of the documents...")
    t4 = time.time()
    all_docs_k_shingles = k_shingles()
    t5 = time.time()
    print("Representing documents with k-shingles took", t5 - t4, "sec\n")

    # signatures sets
    print("Starting to create the signatures of the documents...")
    t6 = time.time()
    signature_sets = signature_set(all_docs_k_shingles)
    t7 = time.time()
    print("Signatures representation took", t7 - t6, "sec\n")
    print(signature_sets)
    print(signature_sets.shape)

    # Permutations
    print("Starting to simulate the MinHash Signature Matrix...")
    t8 = time.time()
    min_hash_signatures = min_hash(signature_sets)
    t9 = time.time()
    print("Simulation of MinHash Signature Matrix took", t9 - t8, "sec\n")
    print(min_hash_signatures)

    # LSH
    print("Starting the Locality-Sensitive Hashing...")
    t10 = time.time()
    candidate_docs = lsh(min_hash_signatures)
    t11 = time.time()
    print("LSH took", t11 - t10, "sec\n")

    # Candidate similarities
    print("Starting to calculate similarities of the candidate documents...")
    t12 = time.time()
    lsh_similarity_matrix = candidates_similarities(candidate_docs, min_hash_signatures)
    t13 = time.time()
    print("Candidate documents similarity calculation took", t13 - t12, "sec\n\n")

    # Return the over t similar pairs
    print(
        "Starting to get the pairs of documents with over ",
        parameters_dictionary["t"],
        "% similarity...",
    )
    t14 = time.time()
    pairs = return_results(lsh_similarity_matrix)
    t15 = time.time()
    print("The pairs of documents are:\n")
    for p in pairs:
        print(p)
    print("\n")

    # Count false negatives and positives
    if parameters_dictionary["naive"]:
        print("Starting to calculate the false negatives and positives...")
        t16 = time.time()
        false_negatives, false_positives = count_false_neg_and_pos(
            lsh_similarity_matrix, naive_similarity_matrix
        )
        t17 = time.time()
        print(
            "False negatives = ",
            false_negatives,
            "\nFalse positives = ",
            false_positives,
            "\n\n",
        )

    if parameters_dictionary["naive"]:
        print("Naive similarity calculation took", t3 - t2, "sec")

    print("LSH process took in total", t13 - t4, "sec")
