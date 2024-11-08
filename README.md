# babble-etl

# Data Processing and Analysis Tool Documentation

### Overview

This documentation outlines the process of data extraction, cleaning, transformation, and analysis of a CSV file using a configuration file in JSON format.


## JSON Configuration Structure
Below is an example of a JSON configuration file structure:

{
    "data_file": "",
    "columns": [
        
    ],
    "transformations": {

    },
    "fill_na": {

    },
    "rename_columns": {

    },
    "data_types": {

    }
}


## Key Fields in the JSON File
    * data_file: Path to the raw CSV file to be processed.
    * columns: List of columns to be included in the dataset.
    * transformations: Dictionary specifying any transformations to be applied to the data.
    * fill_na: Dictionary for handling missing values in specific columns.
    * rename_columns: Dictionary mapping original column names to new names.
    * data_types: Dictionary specifying data types for columns, if applicable.


## Command Line Arguments

The following command-line arguments can be used to run the script:

| Argument         | Type   | Required | Default | Description                                                                                   |
|------------------|--------|----------|---------|-----------------------------------------------------------------------------------------------|
| -i, --input      | String | Yes      | N/A     | Path to the input JSON file for data cleaning and transformation steps.                       |
| -m, --minlength  | Int    | No       | 2       | Minimum length for sequences used in pair analysis.                                           |
| -k, --kmeans     | Int    | No       | 6       | Number of clusters for k-means clustering.                                                    |
| -a, --analysis   | String | No       | N/A     | Type of frequency analysis to perform (choices: singles, pairs, triples, quads, quints, all). |
| -d, --dump       | Flag   | No       | False   | Flag to indicate if sequences should be dumped into a plot.                                   |
| -l, --loglevel   | String | No       | WARNING | Log level for script execution (choices: DEBUG, INFO, WARNING, ERROR, CRITICAL).              |


## Script Workflow

1. **Configuration Parsing:**
    * The script reads the JSON configuration file specified by the -i argument.
2. **Data Extraction:**
    * The CSV file specified in data_file is loaded.
3. **Data Cleaning and Transformation:**
    * The script applies column selection, renaming, transformations, and missing value handling as defined in the JSON configuration.
4. **Analysis:**
    * Depending on the --analysis argument, the script performs sequence analysis, including singles, pairs, triples, quads, or quints.
5. **Logging:**
    * Log messages are configured based on the --loglevel argument for better tracking and debugging.
6. **Dumping Sequences:**
    * If --dump is specified, the processed sequences are saved for further inspection.


## Example Usage

Run the script using the following command:

```bash
python script_name.py -i config.json -m 3 -k 5 -a pairs -l INFO --dump
```

### Explanation:
    * Reads configuration from config.json.
    * Minimum sequence length is set to 3.
    * Runs k-means clustering with 5 clusters.
    * Performs pair analysis.
    * Logs messages at INFO level.
    * Dumps sequences into a plot.


## Logging

Logs are configured to display timestamps, log levels, and messages to help track the script's progress and troubleshoot any issues.

```python
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
```


## Return Value

The script does not return a value but writes logs and, if specified, dumps sequence outputs for plotting.