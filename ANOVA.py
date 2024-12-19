import pandas as pd
import ast
import gc
import os
import subprocess

from mpi4py import MPI
from itertools import combinations
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from multiprocessing import Pool, cpu_count

# MPI comm declarations
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

CHUNKSIZE = 50


def clean_and_prepare_data(chunk):
    """
    Retrieve data from csv file to clean and prepare the entire dataset
    :param chunk: The dataset
    :return: The dataset that are clean  and prepare

    TODO: Fix the date columns to datetime, ANOVA can only read in certain formate
    """

    # Convert date columns to datetime (vectorized)
    # date_columns = ['Hatch date', 'Fledge date', 'Date on vocalization']
    # for col in date_columns:
    #     if col in chunk.columns:
    #         chunk[col] = pd.to_datetime(chunk[col], errors='coerce')

    # Replace spaces with underscores in column names
    chunk.columns = chunk.columns.str.replace(" ", "_")

    # Extract statistics from 'Babbles' column (optimized with vectorized processing)
    if "Babbles" in chunk.columns:

        def process_babbles_vectorized(babbles):
            try:
                babble_list = ast.literal_eval(babbles)
                if isinstance(babble_list, list):
                    return (
                        len(babble_list),
                        sum(babble_list) / len(babble_list),
                        sum(babble_list),
                    )
                else:
                    return 0, 0, 0
            except (ValueError, SyntaxError):
                return 0, 0, 0

        babble_stats = chunk["Babbles"].apply(process_babbles_vectorized)
        chunk[["Babble_Length", "Babble_Mean", "Babble_Sum"]] = pd.DataFrame(
            babble_stats.tolist(), index=chunk.index
        )

    # Rename columns
    chunk = chunk.rename(
        columns={
            "Bout_no.": "Bout_number",
            # 'No._eggs_hatched_from_nest': 'Number_eggs_hatched_from_nest',
            # 'No._birds_fledged_from_nest': 'Number_birds_fledged_from_nest'
        }
    )

    return chunk


def get_header_combinations(csv_file, exclude_headers=[]):
    """
    Retrieve data from csv file to extract headers that will used and some to exclued
    :param csv_file: The path to the csv_file
    :param exclude_headers: A list of headers to remove if needed
    :return: A list of combinations to preform ANOVA Testing
    """
    df = pd.read_csv(csv_file, nrows=0)  # Only reads headers
    headers = df.columns.str.replace(" ", "_").tolist()

    filtered_headers = [header for header in headers if header not in exclude_headers]

    # Precompute all header combinations
    all_combinations = [
        comb
        for r in range(1, len(filtered_headers) + 1)
        for comb in combinations(filtered_headers, r)
    ]
    return all_combinations


def run_anova_parallel(args):
    """
    Run the ANOVA in parallel
    Retrieves info need to format the ANOVA Formula to run Testing
    :param args: A list of headers to remove if needed
    :return: The Anova Result from testing
    """
    chunk, combination, response_col = args
    try:
        data = chunk[list(combination) + [response_col]].dropna()
        if data.empty:
            return None

        # Construct formula and run ANOVA
        formula = f"{response_col} ~ " + " * ".join(combination)
        model = ols(formula, data=data).fit()
        anova_result = anova_lm(model)
        anova_result["Combination"] = str(combination)
        return anova_result
    except Exception as e:
        print(f"Error with combination {combination}: {e}")
        return None


def run_anova(chunk, combinations, response_col):
    for combo in combinations:
        try:
            column_names = chunk[list(combo) + [response_col]]
        except KeyError:
            # skip combinations with missing columns
            continue

        # check if there's enough data in the columns
        if column_names.isnull().sum().sum() > 0:
            print(f"Skipping combination {combo} due to missing data")
            continue

        # construct ANOVA formula
        formula = f"{response_col} ~ " + " * ".join(combo)

        try:
            model = ols(formula, data=column_names).fit()
            anova_result = anova_lm(model)

            anova_result["Combination"] = str(combo)

            return anova_result
        except Exception as e:
            print(f"Error with combination {combo}: {e}")


def process_csv(
    csv_file, header_combinations, response_col="Babble_Length", chunksize=50000
):
    """
    Retrieves all vars needed to get results from run_anova_parallel to Process CSV in chunks
    :param csv_file: Path to CVS File
    :param header_combinations: A list of combinations of headers
    :param response_col: Response Col for testing (Dependent Var)
    :param chunksize: Size of datasets being
    :return: CSV file of the Anova Result
    """
    results = []
    chunk_iter = pd.read_csv(csv_file, chunksize=chunksize)

    for chunk in chunk_iter:
        chunk = clean_and_prepare_data(chunk)

        # Prepare arguments for parallel processing
        args = [(chunk, combo, response_col) for combo in header_combinations]

        # Use multiprocessing for ANOVA
        with Pool(cpu_count()) as pool:
            partial_results = pool.map(run_anova_parallel, args)
            results.extend([res for res in partial_results if res is not None])

        gc.collect()

    # Save all results at once
    pd.concat(results).to_csv("partial_anova_results.csv", index=False)


def process_csv_MPI(
    chunk, header_combinations, response_col="Babble_Length", chunksize=50000
):
    """
    Called on worker ranks
    Retrieves all cars needed to get results from run_anova to process a chunk of a CSV file
    :param chunk: a portion of an input CSV
    :param header_combinations: A list of combinations of headers
    :param response_col: Response col for testing (dependent var)
    :param chunksize: Size of datasets
    :return: CSV file of the ANOVA result
    """
    chunk = clean_and_prepare_data(chunk)

    # args = [(chunk, combo, response_col) for combo in header_combinations]
    partial_results = run_anova(chunk, header_combinations, response_col)
    pass


def filter_significant_results(
    file="partial_anova_results.csv", output_file="filtered_file.csv"
):
    """
    Retrieves data in CSV file to filter (keep) significant results
    :param file: Path to CVS File
    :param output_file: Path to Output CVS File
    :return: Filter CVS File
    """
    # Filter rows where PR(>F) is less than or equal to 0.05
    df = pd.read_csv(file)
    df_filtered = df[df["PR(>F)"].notna() & (df["PR(>F)"] <= 0.05)]

    df_filtered.to_csv(output_file, index=False)
    print(f"\nSignificant ANOVA results saved to '{output_file}'")


def MPI_processing(header_combinations):
    response_col = "Babble_Length"

    while True:
        COMM.send(RANK, dest=0)  # tell root where to send chunk data
        chunk = COMM.recv(source=0)

        if isinstance(chunk, pd.DataFrame):
            print(f"R{RANK} is cleaning chunk data")
            chunk = clean_and_prepare_data(chunk)
            # args = [(chunk, combo, response_col) for combo in header_combinations]

            # work loop
            partial_results = run_anova(chunk, header_combinations, response_col)

            # write partial results to disk
            # pd.concat(partial_results).to_csv(f"results_{RANK}.csv", index=False)
            partial_results.to_csv(f"results_{RANK}.csv", mode="a", header=False)

        else:
            # received stop message (string) from R0, stop work
            print(f"R{RANK} has no more work to do")
            COMM.send("done", dest=0)
            break


def main():
    if RANK == 0:
        print(f"running with {SIZE} ranks")
        print("Rank 0 is preprocessing...")
        csv_file = "CMBabble_Master_combined.csv"
        exclude_headers = [
            "Babbles, Bout_ID",
            "Notes",
            "Raven work",
            "Date_on_vocalization_2",
            "",
        ]

        try:
            os.remove("merged_results.csv")
        except:
            pass

        subprocess.run(["rm results_*.csv"], shell=True)
        # Precompute header combinations
        header_combinations = get_header_combinations(csv_file, exclude_headers)

        chunk_iter = pd.read_csv(csv_file, chunksize=CHUNKSIZE)
        # print(f"length of chunk_iter: {sum(1 for _ in chunk_iter)}")

        # let other ranks know pre-processing is finished; they ask for work
        print("R0 sending header combinations to all worker ranks")
        for i in range(1, SIZE):
            COMM.send(header_combinations, dest=i)

        pre_responses = 0
        while pre_responses != (SIZE - 1):
            response = COMM.recv(source=MPI.ANY_SOURCE)
            if response == "combinations received!":
                pre_responses += 1

        print("R0 dispatching chunks...")
        for i in range(1, SIZE):
            COMM.send("request work", dest=i)

        pre_responses = 0
        while pre_responses != (SIZE - 1):
            response = COMM.recv(source=MPI.ANY_SOURCE)
            if response == "ready":
                pre_responses += 1

        # dispatch each chunk to worker ranks
        while (x := next(chunk_iter, None)) is not None:
            source = COMM.recv(source=MPI.ANY_SOURCE)
            COMM.send(x, dest=source)

        # when all chunks are sent, let worker ranks know they can stop
        for i in range(1, SIZE):
            COMM.send("no more chunks", dest=i)

        print(f"R0 sent stop messages to {SIZE - 1} workers")
        current_responses = 0
        while current_responses != (SIZE - 1):
            response = COMM.recv(source=MPI.ANY_SOURCE)
            if response == "done":
                current_responses += 1
                print(f"R0 heard from {current_responses} workers")

        # read combinations written by worker ranks and aggregate to one file
        print("combine here todo")
        print("R0 is combining results")
        subprocess.run(["cat results_*.csv >> merged_results.csv"], shell=True)

    else:
        header_combinations = COMM.recv(source=0)  # get header_combinations
        COMM.send("combinations received!", dest=0)

        COMM.recv(source=0)  # wait for rank 0 to finish pre-processing
        COMM.send("ready", dest=0)
        MPI_processing(header_combinations)


if __name__ == "__main__":
    main()
    # csv_file = "CMBabble_Master_clean.csv"
    # csv_file = "CMBabble_Master_combined.csv"
    # exclude_headers = [
    #     "Babbles",
    #     "Bout_ID",
    #     "Notes",
    #     "Raven work",
    #     "Date_on_vocalization_2",
    #     "",
    # ]
    # response_col = "Babble_Length"
    #
    # # Precompute header combinations
    # header_combinations = get_header_combinations(csv_file, exclude_headers)
    #
    # # Process the file and run ANOVA
    # process_csv(csv_file, header_combinations, response_col)
    #
    # # Filter significant results
    # filter_significant_results(
    #     file="partial_anova_results.csv", output_file="filtered_file.csv"
    # )
