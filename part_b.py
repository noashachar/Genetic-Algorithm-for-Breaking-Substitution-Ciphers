from part_a import *


# lowering this value so the algorithm terminates within the lifetime of our sun
PLATEAU_TOLLERANCE = 50
N = 5


def make_local_optimizations(dna, fitness, ciphertext):
    for _ in range(N):
        i = random.randrange(len(dna.genes))
        j = random.randrange(len(dna.genes))

        dna.genes[i], dna.genes[j] = dna.genes[j], dna.genes[i]

        new_fit = calculate_fitness(dna.map_text(ciphertext))

        if new_fit > fitness:
            fitness = new_fit
        elif new_fit < fitness:  # if worse, roll back
            dna.genes[i], dna.genes[j] = dna.genes[j], dna.genes[i]

    return fitness


def print_stats(fitnesses, gen_num, epoch_time, plateau_count):
    print('generation #%-6d %.4fs' % (gen_num, epoch_time))

    fns = [np.max, np.min, np.average]
    fn_names = ['max', 'min', 'avg']
    for (fn, fn_name) in zip(fns, fn_names):
        print('fitness-%-10s %.4f' % (fn_name, fn(fitnesses)))

    print(f'plateau count {plateau_count} (will halt at {PLATEAU_TOLLERANCE})', end='\n\n')


# this takes ~[46-53] generations to get to the solution
def lamarck_solve(ciphertext: str, verbose=False) -> DNA:
    ciphertext = ciphertext.lower()

    population = np.array([DNA.new_random() for _ in range(POPULATION_SIZE)])

    best_fitness_ever = -np.Infinity
    plateau_count = np.nan
    epoch_start = now()

    for gen_num in itertools.count():

        fitnesses = calculate_fitnesses(ciphertext, population)

        for i, dna in enumerate(population):
            fitnesses[i] = make_local_optimizations(
                dna, fitnesses[i], ciphertext)

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
    print('num_generations * (POPULATION_SIZE * (N + 1)) ==',
          gen_num * (POPULATION_SIZE * (N + 1)))

    return population[np.argmax(fitnesses)]


# this takes ~[115] generations to get to the solution
def darwin_solve(ciphertext: str, verbose=False) -> DNA:
    ciphertext = ciphertext.lower()

    population = np.array([DNA.new_random() for _ in range(POPULATION_SIZE)])

    best_fitness_ever = -np.Infinity
    plateau_count = np.nan
    epoch_start = now()

    for gen_num in itertools.count():

        fitnesses = calculate_fitnesses(ciphertext, population)

        original_population = np.array([dna.exact_clone() for dna in population])
        for i, dna in enumerate(population):
            fitnesses[i] = make_local_optimizations(
                dna, fitnesses[i], ciphertext)
        population = original_population

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
    print('num_generations * (POPULATION_SIZE * (N + 1)) ==',
          gen_num * (POPULATION_SIZE * (N + 1)))

    return population[np.argmax(fitnesses)]




#########################################################################
###                              m a i n                              ###
#########################################################################



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


def ask_darwin_or_lamarck():
    method = input('darwin or lamarck? ').strip().lower()
    while method not in { 'darwin', 'lamarck' }:
        method = input("please type 'darwin' or 'lamarck': ").strip().lower()
    return method
    

def main():
    is_verbose = parse_argv()

    cipher_filename, ciphertext = ask_for_ciphertext()

    method = ask_darwin_or_lamarck()

    print(f'using {method} to solve {cipher_filename} | is_verbose =', is_verbose, end='\n\n')

    solver_fns = { 'lamarck': lamarck_solve, 'darwin': darwin_solve }
    mapping = solver_fns[method](ciphertext, is_verbose)

    print('final result:', mapping.map_text(ciphertext), sep='\n\n')

    print('substitution table:', mapping.to_simple_columns_string(), sep='\n\n')
    


if __name__ == '__main__':
    main()
