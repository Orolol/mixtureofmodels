# Mixture of Experts (MoE) Model for Instruction Classification and Execution

This project implements a Mixture of Experts (MoE) model for classifying and executing various types of instructions using multiple specialized models. The system is designed to efficiently handle a wide range of tasks by routing instructions to the most appropriate model.

## Project Structure

The project is organized into two main components:

1. Dataset Building (`dataset_build/`)
2. Mixture Training (`mixture_training/`)

### Dataset Building

The dataset building component is responsible for creating and managing the instruction dataset, generating responses from multiple models, and evaluating those responses.

Key files:
- `dataset_build/src/builder.py`: Main script for dataset creation and management
- `dataset_build/src/data_loader.py`: Loads datasets from various sources
- `dataset_build/src/model_loader.py`: Loads different models for instruction processing
- `dataset_build/src/evaluator.py`: Evaluates model responses
- `dataset_build/src/dataset_builder.py`: Manages the dataset structure

### Mixture Training

The mixture training component focuses on training the instruction classifier and implementing the MoE controller.

Key files:
- `mixture_training/src/training.py`: Main script for training the MoE model
- `mixture_training/src/instruction_classifier.py`: Implements the instruction classifier using transformer models
- `mixture_training/src/xgb_classifier.py`: Alternative classifier using XGBoost
- `mixture_training/src/model_recommender.py`: Recommends the best model for a given instruction
- `mixture_training/src/moe_controller.py`: Implements the MoE controller
- `mixture_training/src/model_executor.py`: Executes instructions using the selected model
- `mixture_training/src/llm_classifier.py`: LLM-based classifier for instructions

## Features

- Multi-model instruction processing
- Transformer-based instruction classification (RoBERTa, BERT)
- XGBoost-based instruction classification
- LLM-based instruction classification
- Model recommendation system
- Mixture of Experts (MoE) controller for efficient task routing
- Extensible architecture for adding new models and instruction types

## Usage

1. Dataset Building:
   ```
   python dataset_build/src/builder.py --use_existing_dataset False --batch_size 20 --max_iter 50000
   ```

2. Mixture Training:
   ```
   python mixture_training/src/training.py --epochs 10 --batch_size 16 --model_type roberta-large
   ```

## Requirements

- Python 3.7+
- PyTorch
- Transformers (Hugging Face)
- XGBoost
- Pandas
- NumPy
- Scikit-learn
- NLTK

## Future Improvements

- Implement more sophisticated model selection algorithms
- Add support for multi-task learning
- Improve the feedback mechanism for continuous model updating
- Expand the range of supported instruction types and models

## Contributing

Contributions to improve the project are welcome. Please follow the standard GitHub pull request process to submit your changes.

## License

[MIT License](LICENSE)
