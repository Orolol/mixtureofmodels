# Mixture of Models - Dataset Generation

This project is the first part of a larger initiative to create a "mixture of models" system. It focuses on generating a dataset that contains instructions, responses from various models, and their evaluations.

## Project Structure

```
project_root/
│
├── config/
│   └── config.yaml
│
├── src/
│   ├── data_loader.py
│   ├── model_loader.py
│   ├── evaluator.py
│   ├── dataset_builder.py
│   └── main.py
│
├── requirements.txt
└── README.md
```

## Setup

1. Clone this repository.
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Update the `config/config.yaml` file with your specific datasets, models, and parameters.

## Running the Project

To generate the dataset, run:

```
python src/main.py
```

This will create a CSV file in the `output/` directory containing the generated dataset.

## Configuration

The `config.yaml` file contains all the necessary configurations for datasets, models, and evaluation parameters. Modify this file to change the behavior of the dataset generation process.

## Components

- `data_loader.py`: Loads datasets from Hugging Face.
- `model_loader.py`: Loads and wraps LLM models.
- `evaluator.py`: Evaluates model responses.
- `dataset_builder.py`: Constructs the final dataset.
- `main.py`: Orchestrates the entire process.

## Output

The final dataset will be saved as a CSV file, containing instructions, responses from each model, and their respective evaluation scores.
