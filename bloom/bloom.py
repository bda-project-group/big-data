# This is the code for the Bloom Filter project of TDT4305

import configparser  # for reading the parameters file
from pathlib import Path  # for paths of files
import time  # for timing
import numpy as np  # for creating the bit array
import numpy.typing as npt
import random
from typing import Callable, Optional, Any

# Global parameters
parameter_file = "default_parameters.ini"  # the main parameters file
data_main_directory = Path("data")  # the main path were all the data directories are
parameters_dictionary = (
    dict()
)  # dictionary that holds the input parameters, key = parameter name, value = value


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
            else:
                parameters_dictionary[key] = int(config[section][key])


BitArray = npt.NDArray[np.bool_]

HashFunction = Callable[[str], int]


# TASK 2
# Signature updated to accept the bit array and hash functions, allowed as per Piazza@135.
def bloom_filter(
    new_password: str,
    bit_array: BitArray,
    hash_functions: list[HashFunction],
    update_occurances: bool = True,
) -> np.bool_:
    index = [hash_function(new_password) for hash_function in hash_functions]
    seen = np.all(bit_array[index])

    if update_occurances:
        bit_array[index] = 1

    return seen


# DO NOT CHANGE THIS METHOD
# Reads all the passwords one by one simulating a stream and calls the method bloom_filter(new_password)
# for each password read
# Signature updated to accept the bit array and hash functions for the bloom filter, allowed as per
# Piazza@135.
def read_data(file, bit_array: BitArray, hash_functions: list[HashFunction]):
    time_sum = 0
    pass_read = 0
    with file.open() as f:
        for line in f:
            pass_read += 1
            new_password = line[:-3]
            ts = time.time()
            bloom_filter(new_password, bit_array, hash_functions)
            te = time.time()
            time_sum += te - ts

    return pass_read, time_sum


def _miller_rabin_test(n: int, k: int = 100) -> bool:
    """
    Miller-Rabin test
    Derived from pseudocode in
    [Miller-Rabin test](https://en.wikipedia.org/wiki/Miller-Rabin_primality_test).

    Arguments:
        n: int
            An odd integer > 2 to be tested for primality
        k: int
            The number of rounds of testing to perform
    Returns:
        bool
            True if n is probable prime, False if n is _definitely_ composite
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
    Returns:
        int
            The next prime number
    """
    random.seed(seed)

    assert n > 2, "n must be greater than 2"
    if (n % 2) == 0:
        n += 1

    while not (_miller_rabin_test(n)):
        n += 2
    return n


# TASK 1
# Created h number of hash functions
# Name of function changed from `hash_functions` to avoid name collisions with variables
def create_hash_functions() -> list[HashFunction]:
    random.seed(42)

    base_integers = [
        random.randint(2, np.iinfo("int").max) for _ in range(parameters_dictionary["h"])
    ]
    primes = [_next_prime(base_integer) for base_integer in base_integers]

    hash_functions = []

    for prime in primes:

        def _hash(s: str) -> int:
            return sum(ord(c) * prime**i for i, c in enumerate(s)) % parameters_dictionary["n"]

        hash_functions.append(_hash)

    return hash_functions


passwords_not_in_passwords_csv_file = [
    "07886819",
    "0EDABE59B",
    "0BFE0815B",
    "01058686E",
    "09DFB6D3F",
    "0F493202C",
    "0CA5E8F91",
    "0C13EC1D9",
    "05EF96537",
    "03948BA8F",
    "0D19FB394",
    "0BF3BD96C",
    "0D3665974",
    "0BBDF91E9",
    "0A6083B64",
    "0D76EF8EC",
    "096CD1830",
    "04000DE73",
    "025C442BA",
    "0FD6CAA0A",
    "06CC18905",
    "0998DDE00",
    "02BAACDC4",
    "0D58264FC",
    "0CB8911AA",
    "0CF9E0BDC",
    "007B7F82F",
    "0948FD17A",
    "058BB08DB",
    "02EDBE8CA",
    "0D6F02EFD",
    "09C9797FB",
    "0F8CB3DA5",
    "0C2825430",
    "038BE7E61",
    "03F69C0F5",
    "07EB08903",
    "0917C741D",
    "0D01FEE8F",
    "01B09A600",
    "0BD197525",
    "06B6A2E60",
    "0B72DEF61",
    "095B17373",
    "0B6E0EEB1",
    "0078B3053",
    "08BD9D53F",
    "01995361F",
    "0F0B50CAE",
    "0B5D2887E",
    "004EB658C",
    "0D2C77EDB",
    "07221E24D",
    "0E8A4CC90",
    "00E947367",
    "0DBE190BB",
    "0D8726592",
    "06C02D59D",
    "0462B8BC6",
    "0F85122F8",
    "0FA1961EB",
    "035230553",
    "04CDFB216",
    "0356DB0AD",
    "0FD947DA3",
    "053BB206F",
    "0D1772CC1",
    "00DB759F5",
    "072FB4E7A",
    "0B47CB62D",
    "0616B627F",
    "0F3E153BC",
    "0F3AC7DEE",
    "01286192B",
    "009F3C478",
    "07D89E83E",
    "007CAFDE6",
    "0ABC9E80B",
    "091D1CDA5",
    "0BFC208A1",
    "0957D4C84",
    "00AAF260A",
    "09CF00D7C",
    "0D1C66C72",
    "0EA20CA23",
    "07D6BE324",
    "05B264527",
    "0D48C41F6",
    "081E31BF5",
    "0A1DC7455",
    "07BB493D8",
    "050036F1B",
    "00E73A1EC",
    "0C2D93CC0",
    "0FF47B30C",
    "0313062DE",
    "0E1BEFA3F",
    "0A24D069F",
    "02A984386",
    "0367F7405",
]


if __name__ == "__main__":
    # Reading the parameters
    read_parameters()

    # Creating the hash functions
    hash_functions = create_hash_functions()

    hash_functions: list[HashFunction] = []
    bit_array: BitArray = np.zeros(parameters_dictionary["n"], dtype=np.bool_)

    # Reading the data
    print("Stream reading...")
    data_file = (data_main_directory / parameters_dictionary["data"]).with_suffix(".csv")
    passwords_read, times_sum = read_data(data_file, bit_array, hash_functions)
    print(
        passwords_read,
        "passwords were read and processed in average",
        times_sum / passwords_read,
        "sec per password\n",
    )

    false_positives = 0

    for password in passwords_not_in_passwords_csv_file:
        if bloom_filter(password, bit_array, hash_functions, update_occurances=False):
            false_positives += 1

    print(f"Out of 100 new, unseen passwords, we had {false_positives} false positives.")

    k = parameters_dictionary["h"]
    n = parameters_dictionary["n"]
    false_pos_probability = (1 - np.exp(-k * passwords_read / n)) ** k
    print("False Positive Probability: ", false_pos_probability)
