import json
import math
import random
import numpy as np
from time import time as now
import itertools
from sys import argv


def round_to_even_number(x):
    return round(x / 2) * 2


#########################################################################
###                      h y p e r   p a r a m s                      ###
#########################################################################


# this many genes (out of 26) will come from parent 1
MERGE_PARENT_1_GENES = 5

# we'll have this many dna's competing every generation.
POPULATION_SIZE = round_to_even_number(500)

# this many places in the next generation are for the fittest dna's.
ELITISM_SIZE = round_to_even_number(POPULATION_SIZE * 0.15)

# this many places in the next generation are for crossovers between existing dna's.
CROSSOVER_SIZE = POPULATION_SIZE - ELITISM_SIZE

# every new crossover will have this chance to mutate.
MUTATION_CHANCE = 0.5

# if in the last this many generations, max fitness hasn't increased, halt.
PLATEAU_TOLLERANCE = 100

# the larger, the better chances the fitter will have.
TOURNAMENT_SIZE = 5

# β controlls the exponential decay on the fitnesses in a tournament.
# 0.0 == no decay: the chances to win are proportional to the fitness.
# 1.0 == decay at max: the tournament's fittest will always win.
# β can also be negative, to boost the chances of the less successful.
β = .15

# the tournament's competitors are sorted by fitness,
# and the bad one are further punished,
# as their fitness will then be multiplied by a small number.
EXPONENTIAL_DECAY = np.array([
    (1 - β) ** i
    for i in range(TOURNAMENT_SIZE)
])


#########################################################################
###                           f i t n e s s                           ###
#########################################################################


def parse_frequencies_file():
    with open('freq-triplets.json') as file:
        return {tuple(letters): math.log(freq) for letters, freq in json.load(file).items()}


triplets_score = parse_frequencies_file()


def triplet_wise(text):
    """
    '123456' -> tuple('123'), tuple('234'), tuple('345'), tuple('456')
    """
    return zip(text, text[1:], text[2:])
    

def calculate_fitness(text):
    """
    sums the likelyhood-score of every letters-triplet in the text.
    """
    return sum(
        triplets_score[triplet]
        for triplet in triplet_wise(text)
        if triplet in triplets_score
    )


#########################################################################
###                               d n a                               ###
#########################################################################


letters = 'abcdefghijklmnopqrstuvwxyz'


class DNA:
    def __init__(self, genes=None):
        """
        `genes` is some permuation of the letters [a-z]. no letter can appear twice.
        """
        self.genes = genes

    @staticmethod
    def new_random():
        genes = [*letters]
        random.shuffle(genes)
        return DNA(genes)

    def __repr__(self) -> str:
        return f"DNA({self.genes})"
    
    def to_simple_columns_string(self):
        """
        returns something like this
        ```
            a o
            b a
            c d
        ```
        """
        results = ''
        for letter, substitution in zip(letters, self.genes):
            results += f"{letter} {substitution}" + '\n'
        return results

    def to_pretty_string(self):
        """
        returns something like this
        ```
            |-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
            |  a  |  b  |  c  |  d  |  e  |  f  |  g  |  h  |  i  |  j  |  k  |
            |-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
            |  f  |  c  |  o  |  h  |  k  |  q  |  b  |  p  |  l  |  e  |  g  |
            |-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
        ```
        """
        header = "|-----" * len(letters) + "|"
        row_format = "|  {}  " * len(letters) + "|"
        lines = [
            header,
            row_format.format(*letters),
            header,
            row_format.format(*self.genes),
            header,
            ''
        ]
        return '\n'.join(lines)


    def map_text(self, text):
        # Convert text to NumPy array of character codes
        text_codes = np.frombuffer(text.encode(), np.uint8)

        # Create a mapping array
        mapping = np.array(self.genes, dtype='S1').view(np.uint8)

        # Create the cipher array
        cipher = np.copy(text_codes)

        # Apply the substitution cipher using NumPy indexing
        mask = (text_codes >= ord('a')) & (text_codes <= ord('z'))
        cipher[mask] = mapping[text_codes[mask] - ord('a')]

        # Convert back to string
        return cipher.tobytes().decode()

    def mutate_in_place(self):
        i = random.randrange(len(self.genes))
        j = random.randrange(len(self.genes))
        self.genes[i], self.genes[j] = self.genes[j], self.genes[i]

    def exact_clone(self):
        return DNA(self.genes.copy())

    @staticmethod
    def a_crossover_between(dna_1, dna_2):
        merged_genes = dna_2.genes.copy()

        parent_1_indices = random.sample(range(len(dna_1.genes)), k=MERGE_PARENT_1_GENES)

        new = set(map(dna_1.genes.__getitem__, parent_1_indices))
        old = set(map(dna_2.genes.__getitem__, parent_1_indices))

        gone = old - new  # letters that will be lost
        dups = new - old  # letters that we'll have more than once

        # assign some of parent_1's genes into the copy of parent_2's genes
        for idx in parent_1_indices:
            merged_genes[idx] = dna_1.genes[idx]

        # fix the dups/gone genes
        for idx, letter in enumerate(dna_2.genes):
            if letter in dups:
                merged_genes[idx] = gone.pop()
                dups.remove(letter)
                if not dups:
                    break

        return DNA(merged_genes)

        # example of a crossover
        #
        # dna_1.genes 7 5 8 6 0 9 1 4 3 2
        #
        # dna_2.genes 0 1 2 3 4 5 6 7 8 9
        #                   |   | |
        # indices [3 5 6]   |   | |
        #                   |   | |
        # merg        0 1 2 6 4 9 1 7 8 9
        #               |       |
        # gone { 3 5 }  |       |
        # dups { 9 1 }  |       |
        # good { 6 }    |       |
        #               |       |
        # result      0 3 2 6 4 5 1 7 8 9



#########################################################################
###                 g e n e t i c   a l g o r i t h m                 ###
#########################################################################



def calculate_fitnesses(cipher: str, population: np.ndarray) -> np.ndarray:
    return np.array([
        calculate_fitness(text) for text in (
            dna.map_text(cipher) for dna in population
        )
    ])


def find_elite(population: np.ndarray, fitnesses: np.ndarray) -> np.ndarray:
    top_indices = np.argsort(-fitnesses)[:ELITISM_SIZE]
    return population[top_indices]


def perform_tournament(population: np.ndarray, fitnesses: np.ndarray) -> np.ndarray:
    some_indices = random.sample(range(POPULATION_SIZE), k=TOURNAMENT_SIZE)
    candidates = population[some_indices]
    scores = fitnesses[some_indices] * EXPONENTIAL_DECAY

    scores_sum = scores.sum()
    if scores_sum > 0:
        probabilities = scores / scores.sum()
    else:
        probabilities = None

    return np.random.choice(candidates, size=2, replace=False, p=probabilities)


def make_crossovers(population: np.ndarray, fitnesses: np.ndarray) -> list:
    results = []
    for _ in range(CROSSOVER_SIZE // 2):
        parent_1, parent_2 = perform_tournament(population, fitnesses)
        for _ in range(2):
            child_dna = DNA.a_crossover_between(parent_1, parent_2)
            if random.random() < MUTATION_CHANCE:
                child_dna.mutate_in_place()
            results.append(child_dna)
    return results


def do_one_epoch(population, fitnesses):
    crossovers = make_crossovers(population, fitnesses)
    elite = find_elite(population, fitnesses)
    next_population = np.concatenate((elite, crossovers))
    return next_population


def print_stats(fitnesses, gen_num, epoch_time, plateau_count):
    print(
        'generation #%-6d' % gen_num if np.isnan(epoch_time) else
        'generation #%-6d %.4fs' % (gen_num, epoch_time))

    fns = [np.max, np.min, np.average, np.std]
    fn_names = ['max', 'min', 'avg', 'std']
    for (fn, fn_name) in zip(fns, fn_names):
        print('fitness-%-10s %.4f' % (fn_name, fn(fitnesses)))

    print(f'plateau count {plateau_count} (will halt at {PLATEAU_TOLLERANCE})', end='\n\n')


def solve(ciphertext: str, verbose=True) -> DNA:
    ciphertext = ciphertext.lower()

    population = np.array([DNA.new_random() for _ in range(POPULATION_SIZE)])

    best_fitness_ever = -np.Infinity
    plateau_count = np.nan
    epoch_start = np.nan

    for gen_num in itertools.count():
        fitnesses = calculate_fitnesses(ciphertext, population)

        best_fitness = max(fitnesses)
        if best_fitness > best_fitness_ever:
            best_fitness_ever = best_fitness
            plateau_count = 0
        else:
            plateau_count += 1

        print_stats(fitnesses, gen_num, now() - epoch_start, plateau_count)

        if verbose:
            idx = np.argmax(fitnesses)
            print(population[idx].map_text(ciphertext), end='\n\n')

        if plateau_count >= PLATEAU_TOLLERANCE:
            break

        epoch_start = now()

        population = do_one_epoch(population, fitnesses)

    print('number of times fitness function was called:')
    print('num_generations * POPULATION_SIZE ==', gen_num * POPULATION_SIZE)

    return population[np.argmax(fitnesses)]



#########################################################################
###                              m a i n                              ###
#########################################################################



PATH_PLAIN_TEXT = 'plain.txt'
PATH_KEY = 'perm.txt'


FLAG_NO_VERBOSE = '--minimal-prints'


def parse_argv():
    redundant_args = set(argv[1:]) - {FLAG_NO_VERBOSE}

    if redundant_args:
        print("warning: the following args will be ignored:", *redundant_args)

    is_verbose = FLAG_NO_VERBOSE not in argv

    return is_verbose


def ask_for_ciphertext():
    DEFAULT_PATH_ENC = 'enc.txt'

    enc_filename = input(f"enter path of ciphertext (leave for '{DEFAULT_PATH_ENC}') > ")
    enc_filename = enc_filename or DEFAULT_PATH_ENC

    try:
        with open(enc_filename) as file:
            ciphertext = file.read()

    except Exception as err:
        print(f"could not read ciphertext file: {err}")
        exit(1)

    return enc_filename, ciphertext


def main():
    is_verbose = parse_argv()

    cipher_filename, ciphertext = ask_for_ciphertext()

    print(f'solving {cipher_filename} | is_verbose =', is_verbose, end='\n\n')

    mapping = solve(ciphertext, is_verbose)

    with open(PATH_PLAIN_TEXT, 'w') as file:
        file.write(mapping.map_text(ciphertext))

    with open(PATH_KEY, 'w') as file:
        file.write(mapping.to_simple_columns_string())

    print()
    print('fittest plain-text was saved to', PATH_PLAIN_TEXT)
    print('permutation (subsitution table) was saved to', PATH_KEY)


if __name__ == '__main__':
    main()
