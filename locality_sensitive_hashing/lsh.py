"""
This is the implementation of the Locality Sensitive Hashing algorithm for the second project in the course
TDT4305 Big Data, spring 2023 at NTNU, by Hermann Mørkrid and Lars Waage.

The module consists of some skeleton code, as part of the handout for the project, prefixed by # DO NOT CHANGE THIS METHOD.

The project consists of 6 implementation tasks, which are marked with # TASK 1, # TASK 2, etc. in the code.

`k_singles` creates the k-shingles of each document and returns a list of them. The shingles are based around
characters, rather than words, as it was implemented prior to the updates to the project description.

`signature_set` creates the signature set representation of the document shingles, stored as a dense matrix of 
indices of the shingles that exist in each document.

`_next_prime` is a helper function for the `min_hash` function, which adds utility to find the next prime number greater
than a given integer. It is based on the Miller-Rabin primality test, which is a probabilistic test for primality.

`_miller_rabin_test` is an implementation of the aforementioned Miller-Rabin primality test.

`min_hash` creates the min-hash representation of the document shingles by simulating permutations of the signature
matrix using a universal hash function of the form `h(x) = (ax + b) % p % r`, where `a` and `b` are random integers
between 0 and `N`, `p` is the next prime number greater than `N`, where `N` is the total number of unique shingles.


`_hash` is a helper function to hash each band of the min-hash matrix into a bucket.

`lsh` finds the candidate pairs of documents that are similar, based on the min-hash matrix and the number of bands
and buckets.

`candidates_similarity` calculates the similarity of the candidate pairs of documents, and returns a similarity matrix
of the documents.

`return_results` returns the resulting candidate pairs of documents.

`count_false_neg_and_pos` counts the number of false negatives and positives in the results compared to the naive 
similarity matrix.
"""

import configparser  # for reading the parameters file
import os  # for reading the input data
import sys  # for system errors and printouts
import time  # for timing
from collections import defaultdict
from itertools import combinations
from pathlib import Path  # for paths of files
from typing import Literal, TypedDict
import math
import random


import numpy as np
import numpy.typing as npt

"""
To improve readability and static code analysis, we've defined a set of type aliases.
"""


class Parameters(TypedDict):
    k: int
    permutations: int
    r: int
    buckets: int
    data: Literal["test", "bbc"] | str
    naive: bool
    t: float


KShingles = list[npt.NDArray[np.str_]]
Candidates = set[tuple[int, int]]
IntArray = npt.NDArray[np.int64]
MinHashMatrix = IntArray
DenseSignatureSet = list[npt.NDArray[np.intp]]
LSHSimilarityMatrix = npt.NDArray[np.float64]


# Global parameters
parameter_file = "default_parameters.ini"  # the main parameters file
data_main_directory = Path("data")  # the main path were all the data directories are
parameters_dictionary: Parameters = {
    "data": "bbc",
    "k": 5,
    "r": 2,
    "t": 0.6,
    "naive": False,
    "permutations": 20,
    "buckets": 10,
}  # dictionary that holds the input parameters, key = parameter name, value = value

# dictionary of the input documents, key = document id, value = the document
document_list: dict[int, str] = dict()


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
def get_triangle_index(i: int, j: int, length: int) -> int:
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
# Calculates the similarities of all the combinations of documents and returns the similarity
# triangular matrix.
def naive():
    docs_Sets = []  # holds the set of words of each document

    for doc in document_list.values():
        docs_Sets.append(set(doc.split()))

    # Using triangular array to store the similarities, avoiding half size and similarities of i==j
    num_elems = int(len(docs_Sets) * (len(docs_Sets) - 1) / 2)
    similarity_matrix: list[float] = [0 for x in range(num_elems)]
    for i in range(len(docs_Sets)):
        for j in range(i + 1, len(docs_Sets)):
            similarity_matrix[get_triangle_index(i, j, len(docs_Sets))] = jaccard(
                docs_Sets[i], docs_Sets[j]
            )

    return similarity_matrix


# METHOD FOR TASK 1
# Creates the k-Shingles of each document and returns a list of them
def k_shingles() -> KShingles:
    """
    Global Parameters:
        k: int
            The length of the shingles to be created.

    Arguments:
        document_list: dict[int, str] (accessed through global variable)
            A dictionary of length M, mapping document IDs to documents of length N_i:
    ```
    {
        1: "abcde", # document 1
        2: "fghij", # document 2
        3: "klmno", # document 3
    }
    ```
    Returns an array of length M, where each element i is a set of shingles of length k.

    For example, with k=3:
    ```
    [
        {"abc", "bcd", "cde"}, # document 1
        {"fgh", "ghi", "hij"}, # document 2
        {"klm", "lmn", "mno"}, # document 3
    ]
    """
    docs_k_shingles: KShingles = []  # holds the k-shingles of each document

    for doc in document_list.values():
        k = parameters_dictionary["k"]
        shingles: set[str] = set()
        for i in range(len(doc) - k + 1):
            shingles.add(doc[i : i + k])

        docs_k_shingles.append(np.array(list(shingles)))
    return docs_k_shingles


def _miller_rabin_test(n: int, k: int = 100) -> bool:
    """
    Miller-Rabin test
    Derived from pseudocode in [Miller-Rabin test](https://en.wikipedia.org/wiki/Miller-Rabin_primality_test).

    Arguments:
        n: int
            An odd integer > 2 to be tested for primality
        k: int
            The number of rounds of testing to perform
    Returns:
        bool
            True if n is _probably_ prime, False if n is _definitely_ composite
    """
    s, d = 0, n - 1
    while d % 2 == 0:
        s += 1
        d //= 2

    for _ in range(k):
        a = random.randint(2, n - 1)
        x = pow(a, d, n)

        if x == 1 or x == n - 1:
            continue

        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == 1:
                return False

            if x == n - 1:
                break
        else:
            return False
    return True


def _next_prime(n: int, seed=42) -> int:
    """
    Finds the next probable prime number after `n`, based on the Miller-Rabin test.

    Arguments:
        n: int
            An odd integer > 2
        seed: int
            The seed to use for the random number generator
    """
    random.seed(seed)

    assert n % 2 == 1, "n must be odd"
    assert n > 2, "n must be greater than 2"

    while not (_miller_rabin_test(n)):
        n += 2
    return n


total_shingles: int = 0


# METHOD FOR TASK 2
# Creates a signatures set of the documents from the k-shingles list
def signature_set(k_shingles: KShingles) -> DenseSignatureSet:
    """
    For a given list of k-shingles per document, creates a dense signature set of the documents.
    Since a boolean matrix representation is [typically sparse](http://infolab.stanford.edu/~ullman/mining/2009/similarity1.pdf),
    we will use a dense representation instead.

    Arguments:
        k_shingles: KShingles
            Takes an input array of length N, where each element j is a set of shingles that appears
            in document j.
    ```
    [
        {shingle1, shingle2, shingle3}, # document1
        {shingle1, shingle2, shingle3}, # document2
        {shingle3, shingle4, shingle5}, # document3
    ]
    ```

    Which has the unique shingles:
    ```
    [shingle1, shingle2, shingle3, shingle4, shingle5]
    ```

    Returns:
        DenseSignatureSet
            A list of length N, where each element i is an array of indices of shingles that appear
            in document i.
    ```
    [
        [0, 1, 2] # document 1
        [0, 1, 2] # document 2
        [2, 3, 4] # document 3
    ]
    ```
    """
    unique_shingles: set[str] = set()
    for shingles in k_shingles:
        unique_shingles.update(shingles)

    global total_shingles
    total_shingles = len(unique_shingles)

    print("Total shingles:", len(unique_shingles))
    dense_signature_set: DenseSignatureSet = []

    # Sets have no ordering guarantee, so we need to sort them
    # for the signature set to be consistent and reproducable.
    ordered_shingles = np.sort(np.array(list(unique_shingles)))

    for document in k_shingles:
        # Determine which shingles appear in the document and return a boolean mask
        mask = np.isin(ordered_shingles, document, assume_unique=True)
        # Get the indices of the shingles that appear in the document
        indices = np.nonzero(mask)[0]
        dense_signature_set.append(indices)

    return dense_signature_set


# METHOD FOR TASK 3
# Creates the minHash signatures after simulation of permutations
def min_hash(signature_set: DenseSignatureSet) -> MinHashMatrix:
    """
    Takes a dense signature set and simulates permutations using random hash functions
    to create a minHash signature matrix.

    The hash function is of the form
    ```
    h(x) = (ax + b) % p
    ```
    Where `a` and `b` are random integers < N and `p` is the next probable prime number > N, where N is the total number of
    unique shingles in all documents.

    The next probable prime number is found using the Miller-Rabin test.

    Global Parameters:
        permutations: int
            The number of permutations to simulate.

    Arguments:
        signature_set: DenseSignatureSet
            A list of length N, where each element i is an array of indices of shingles that appear
            in document i.

    Returns:
        MinHashMatrix
            A matrix of size P x N, where P is the number of permutations and N is the number of
            documents.
            Each item (i, j) is the index of of the first shingle, i, that appears in document, j,
            for a given permutation.
    """
    rng = np.random.default_rng(seed=42)

    number_of_permutations = parameters_dictionary["permutations"]

    min_hash_signatures: MinHashMatrix = np.empty(
        shape=(number_of_permutations, len(document_list)), dtype=np.int64
    )

    # generate unique coefficients for each permutation
    coefficients = rng.choice(
        np.arange(total_shingles), size=(number_of_permutations, 2), replace=False
    )

    # find the next prime number > N
    if total_shingles % 2 == 0:
        prime = _next_prime(total_shingles + 1)
    else:
        prime = _next_prime(total_shingles)

    def universal_hash(x: npt.NDArray[np.intp], p: int, a: int, b: int) -> IntArray:
        """
        Universal Hash Function

        Arugments:
            x: npt.NDArray[np.intp]
                The array of indices of shingles that appear in a document.
            p: int
                The next probable prime number > N, where N is the total number of unique shingles in all documents.
            a: int
                A random integer < N, where N is the total number of unique shingles in all documents.
            b: int
                A random integer < N, where N is the total number of unique shingles in all documents.

        Returns:
            IntArray
                The hash value of the array of indices of shingles that appear in a document.
        """
        return (a * x + b) % p

    for i in range(number_of_permutations):
        for j, rows in enumerate(signature_set):
            min_hash_signatures[i, j] = np.min(
                universal_hash(rows, prime, *coefficients[i])
            )

    return min_hash_signatures


def _hash(band: IntArray, number_of_buckets: int) -> IntArray:
    """
    Generate the hash key for a given column in a band
    it is generated by taking the sum of the column and dividing it by the number of buckets
    """
    hash_key = np.floor_divide(np.sum(band, axis=0), number_of_buckets)
    return hash_key % number_of_buckets


# METHOD FOR TASK 4
# Hashes the MinHash Signature Matrix into buckets and find candidate similar documents
def lsh(m_matrix: MinHashMatrix) -> Candidates:
    """
    For a given P x N matrix, partitions into P / `r` bands, and hashes each column in each band
    to a bucket. Returns a set of candidate pairs of documents believed to be similar, i.e. hashes
    to the same bucket for a given band.

    Buckets are unique accross bands.

    Global Parameters:
        r: int
            The number of rows in each band. Must be a divisor of P, the number of permutations.
        buckets: int
            The number of buckets to hash each band into.

    Arguments:
        m_matrix: MinHashMatrix
            A min-hash matrix of size P x N, where P is the number of permutations and N is the
            number of documents. Each item (i, j) is the index of of the first shingle, i, that
            appears in document, j, for a given permutation. The matrix is the return value of the
            `min_hash` function.

    Returns:
        Candidates
            A set of candidate pairs of documents believed to be similar, i.e. hashes to the same
            bucket for a given band. Each pair is represented as a tuple of column indices in the
            signature matrix.
    """

    rows = parameters_dictionary["r"]
    number_of_buckets = parameters_dictionary["buckets"]
    number_of_bands = int(m_matrix.shape[0] / rows)
    number_of_columns = m_matrix.shape[1]

    # List of candidate pairs of document signatures for checking similarity
    # Stored as pairs of column indices in the signature matrix
    candidates: Candidates = set()

    for band_index in range(number_of_bands):
        # Offset for the rows in the current band
        offset = band_index * rows
        band = m_matrix[offset : offset + rows, :]

        # Hash each column in the band to a bucket
        buckets = np.apply_along_axis(
            func1d=_hash, axis=0, arr=band, number_of_buckets=number_of_buckets
        )

        bucket_dict: dict[np.int64, set[int]] = defaultdict(set)

        for column_index in range(number_of_columns):
            bucket: np.int64 = buckets[column_index]
            bucket_dict[bucket].add(column_index)

        for column_indices in bucket_dict.values():
            if len(column_indices) > 1:
                candidates = candidates.union(combinations(column_indices, r=2))

    return candidates


# METHOD FOR TASK 5
# Calculates the similarities of the candidate documents
def candidates_similarities(
    candidate_docs: Candidates, min_hash_matrix: MinHashMatrix
) -> LSHSimilarityMatrix:
    """
    Calculates the similarity of each pair of candidate documents.

    Global Parameters:
        document_list: list[str]
            The list of documents to be processed.

    Arguments:
        candidate_docs: Candidates
            A set of candidate pairs of documents believed to be similar (results of `lsh`).
        min_hash_matrix: MinHashMatrix
            A min-hash matrix of size P x N where P is the number of permutations and N is the
            number of documents. Each item (i, j) is the index of of the first shingle, i, that
            appears in document, j, for a given permutation. The matrix is the return value of the
            `min_hash` function.

    Returns:
        LSHSimilarityMatrix
            A vector of size `L choose 2` where `L` is the number of documents. Each item is the
            similarity of a pair of candidate documents. The similarity is the number of rows in
            the min-hash matrix where the two documents have the same signature, divided by the
            total number of rows in the min-hash matrix.

            The vector is stored in a lower triangular matrix, where the index of each item is
            calculated using the `get_triangle_index` function.
    """

    similarity_matrix: LSHSimilarityMatrix = np.zeros(
        shape=(math.comb(len(document_list), 2)), dtype=np.float64
    )

    for column_index_1, column_index_2 in candidate_docs:
        column_1 = min_hash_matrix[:, column_index_1]
        column_2 = min_hash_matrix[:, column_index_2]
        number_of_rows = min_hash_matrix.shape[0]

        similarity = (column_1 == column_2).sum() / number_of_rows

        triangle_index = get_triangle_index(
            column_index_1, column_index_2, len(document_list)
        )
        similarity_matrix[triangle_index] = similarity

    return similarity_matrix


# METHOD FOR TASK 6
# Returns the document pairs of over t% similarity
def return_results(lsh_similarity_matrix: LSHSimilarityMatrix) -> list[tuple[int, int]]:
    """
    Finds the document pairs that are above the threshold similarity.

    Global Parameters:
        t: float
            The threshold similarity. The default value is 0.6.

    Arguments:
        lsh_similarity_matrix: LSHSimilarityMatrix
            A vector of size `L choose 2` where `L` is the number of documents. Each item is the
            similarity of a pair of candidate documents. The vector is stored in a lower triangular
            matrix, where the index of each item is calculated using the `get_triangle_index`
            function. This is the return value of the `candidates_similarities` function.

    Returns:
        list[tuple[int, int]]
            A list of document index pairs that are above the threshold similarity.
    """
    threshold = parameters_dictionary["t"]

    # Find the triangle indices of the similarity matrix that are above the
    # threshold
    indices_above_threshold = np.argwhere(lsh_similarity_matrix >= threshold)

    # Find the row and column indices of the similarity matrix from the
    # triangle indices
    row, column = np.unravel_index(
        indices_above_threshold, shape=(len(document_list), len(document_list))
    )

    # Convert the row and column indices to document pairs
    document_pairs = list(zip(row.flatten().tolist(), column.flatten().tolist()))
    return document_pairs


# METHOD FOR TASK 6
def count_false_neg_and_pos(
    lsh_similarity_matrix: LSHSimilarityMatrix,
    naive_similarity_matrix: list[float],
):
    """
    Counts the number of false positives and false negatives in the LSH similarity matrix.

    False positives are defined as pairs of documents that are above the similarity threshold in
    `lsh_similarity_matrix`, but not in `naive_similarity_matrix`.

    False negatives are defined as pairs of documents that are above the similarity threshold in
    `naive_similarity_matrix`, but not in `lsh_similarity_matrix`.

    Global Parameters:
        t: float
            The threshold similarity. The default value is 0.6.

    Arguments:
        lsh_similarity_matrix: LSHSimilarityMatrix
            A vector of size `L choose 2` where `L` is the number of documents. Each item is the
            similarity of a pair of candidate documents. The vector is stored in a lower triangular
            matrix, where the index of each item is calculated using the `get_triangle_index`
            function. This is the return value of the `candidates_similarities` function.

        naive_similarity_matrix: list[float]
            A vector of size `L choose 2` where `L` is the number of documents. Each item is the
            similarity of a pair of candidate documents. The vector is stored in a lower triangular
            matrix, where the index of each item is calculated using the `get_triangle_index`
            function. This is the return value of the `naive` function.

    Returns:
        tuple[int, int]
            A tuple of two integers: the number of false negatives and the number of false
            positives.
    """

    false_negatives = 0
    false_positives = 0

    threshold = parameters_dictionary["t"]

    for i, naive_similarity in enumerate(naive_similarity_matrix):
        lsh_similarity = lsh_similarity_matrix[i]

        if naive_similarity >= threshold and lsh_similarity < threshold:
            false_negatives += 1
        elif naive_similarity < threshold and lsh_similarity >= threshold:
            false_positives += 1

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

    # Permutations
    print("Starting to simulate the MinHash Signature Matrix...")
    t8 = time.time()
    min_hash_signatures = min_hash(signature_sets)
    t9 = time.time()
    print("Simulation of MinHash Signature Matrix took", t9 - t8, "sec\n")

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
