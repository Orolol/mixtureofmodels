import yaml
import uuid
import pandas as pd
import os
from dataset_build.src.data_loader import load_datasets
from dataset_build.src.model_loader import load_models
from dataset_build.src.evaluator import Evaluator
from dataset_build.src.dataset_builder import DatasetBuilder
import torch

def main():
    print("Check if CUDA is available")
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    
    MAX_ITER = 50000
    OUTPUT_DIR = "dataset_build/output"
    BATCH_SIZE = 100  # Number of instructions to process in each batch
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load configuration
    with open('dataset_build/config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Load datasets
    datasets = load_datasets(config['datasets'], MAX_ITER)

    # Initialize dataset builder
    dataset_builder = DatasetBuilder()

    # Load or create instruction dataset
    instruction_file = f"{OUTPUT_DIR}/dataset_instructions.csv"
    if os.path.exists(instruction_file):
        dataset_builder.dataset_df = pd.read_csv(instruction_file)
    else:
        for dataset_name, dataset in datasets:
            dataset_builder.add_entries_from_loaded_dataset(dataset)
        dataset_builder.dataset_df.to_csv(instruction_file, index=False)

    # Load existing responses and evaluations
    responses_file = f"{OUTPUT_DIR}/dataset_responses.csv"
    evaluations_file = f"{OUTPUT_DIR}/dataset_evaluations.csv"
    if os.path.exists(responses_file):
        dataset_builder.responses_df = pd.read_csv(responses_file)
    if os.path.exists(evaluations_file):
        dataset_builder.evaluations_df = pd.read_csv(evaluations_file)

    # Load model configurations
    model_configs = config['models']

    # Load evaluator
    print("Loading evaluator", flush=True)
    evaluator = Evaluator(config['evaluator_model']['name'],
                          config['evaluator_model']['path'],
                          config['evaluator_model']['parameters'])

    # Process instructions in batches
    for start_idx in range(0, len(dataset_builder.dataset_df), BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, len(dataset_builder.dataset_df))
        batch_df = dataset_builder.dataset_df.iloc[start_idx:end_idx]

        print(f"Processing batch {start_idx//BATCH_SIZE + 1}", flush=True)

        for model_idx, model_config in enumerate(model_configs):
            print(f"Loading model: {model_config['name']}", flush=True)
            model = load_models([model_config])[0]
            dataset_builder.add_model(model_idx, model.name, model.parameters)

            for _, row in batch_df.iterrows():
                question_id = row['question_id']
                instruction = row['instruction']

                # Check if response exists
                existing_response = dataset_builder.responses_df[
                    (dataset_builder.responses_df['question_id'] == question_id) &
                    (dataset_builder.responses_df['model_id'] == model_idx)
                ]

                if existing_response.empty:
                    print(f"Generating response for question {question_id} with model {model.name}", flush=True)
                    response = model.generate(instruction)
                    response_id = str(uuid.uuid4())
                    dataset_builder.add_response(response_id, question_id, model_idx, response)
                    dataset_builder.responses_df.to_csv(responses_file, index=False)

                    # Evaluate the response
                    score = evaluator.evaluate(instruction, response)
                    evaluation_id = str(uuid.uuid4())
                    dataset_builder.add_evaluation(evaluation_id, response_id, score)
                    dataset_builder.evaluations_df.to_csv(evaluations_file, index=False)
                else:
                    response_id = existing_response['response_id'].iloc[0]
                    # Check if evaluation exists
                    existing_evaluation = dataset_builder.evaluations_df[
                        dataset_builder.evaluations_df['response_id'] == response_id
                    ]

                    if existing_evaluation.empty:
                        print(f"Evaluating response for question {question_id} with model {model.name}", flush=True)
                        response = existing_response['response'].iloc[0]
                        score = evaluator.evaluate(instruction, response)
                        evaluation_id = str(uuid.uuid4())
                        dataset_builder.add_evaluation(evaluation_id, response_id, score)
                        dataset_builder.evaluations_df.to_csv(evaluations_file, index=False)

            # Unload the model to free up memory
            del model
            torch.cuda.empty_cache()

        # Save progress after each batch
        dataset_builder.save_datasets(f"{OUTPUT_DIR}/dataset")

    # Print statistics
    print_statistics(dataset_builder)

def print_statistics(dataset_builder):
    print("\n--- Execution Statistics ---")

    total_questions = len(dataset_builder.dataset_df)
    print(f"Total number of questions: {total_questions}")

    num_models = len(dataset_builder.models_df)
    print(f"Number of models: {num_models}")

    merged_df = pd.merge(dataset_builder.responses_df, dataset_builder.evaluations_df, on='response_id')
    merged_df = pd.merge(merged_df, dataset_builder.models_df, on='model_id')

    for _, model in dataset_builder.models_df.iterrows():
        model_id = model['model_id']
        model_name = model['model_name']
        model_data = merged_df[merged_df['model_id'] == model_id]
        
        mean_score = model_data['score'].mean()
        num_responses = len(model_data)
        
        print(f"\nModel: {model_name}")
        print(f"  - Number of responses: {num_responses}")
        print(f"  - Average score: {mean_score:.2f}")

    total_responses = len(dataset_builder.responses_df)
    total_evaluations = len(dataset_builder.evaluations_df)
    overall_mean_score = dataset_builder.evaluations_df['score'].mean()

    print(f"\nGlobal statistics:")
    print(f"  - Total number of responses: {total_responses}")
    print(f"  - Total number of evaluations: {total_evaluations}")
    print(f"  - Overall average score: {overall_mean_score:.2f}")

if __name__ == "__main__":
    main()
