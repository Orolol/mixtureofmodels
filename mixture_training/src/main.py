from mixture_training.src.data_processor import load_data, preprocess_data, split_data
from mixture_training.src.instruction_classifier import InstructionClassifier
from mixture_training.src.xgb_classifier import XGBInstructionClassifier
from mixture_training.src.model_recommender import ModelRecommender
from mixture_training.src.moe_controller import MoEController
from mixture_training.src.model_executor import ModelExecutor
from dataset_build.src.model_loader import load_models
import numpy as np
import pickle
import yaml
import os

def main(num_epochs=10, batch_size=16):
    # Load configuration
    # with open('../dataset_build/config/config.yaml', 'r') as file:
    #     config = yaml.safe_load(file)
    
    # check if the processed data pickle file exists
    # if os.path.exists('dataset_build/save/processed_data.pkl'):
    #     with open('dataset_build/save/processed_data.pkl', 'rb') as f:
    #         processed_data = pickle.load(f)
    # else:
    #     # Load and preprocess data
    #     data = load_data('dataset_build/output/dataset_instructions.csv')
    #     print("Data loaded")
    #     processed_data = preprocess_data(data)
        
    #     # Let's save the processed data to a pickle file
    #     if not os.path.exists('dataset_build/save'):
    #         os.makedirs('dataset_build/save')
    #     print("Saving processed data")
    #     with open('dataset_build/save/processed_data.pkl', 'wb') as f:
    #         pickle.dump(processed_data, f)
    
    processed_data = load_data('dataset_build/output/dataset_instructions.csv')
    features, labels = preprocess_data(processed_data)
    
    print(features.shape)
    print(labels.shape)
            
    print("Data preprocessed")
    X_train, X_test, y_train, y_test = split_data(features,labels)
    print("Data split")

    # Train RoBERTa Instruction Classifier
    print("Training RoBERTa Classifier")
    roberta_classifier = InstructionClassifier(num_classes=len(np.unique(labels)))
    roberta_classifier.train(X_train, y_train, num_epochs=num_epochs, batch_size=batch_size)
    
    # Evaluate the RoBERTa model
    y_pred_roberta = roberta_classifier.predict(X_test)
    accuracy_roberta = np.mean(y_pred_roberta == y_test)
    print(f"RoBERTa Test accuracy: {accuracy_roberta:.4f}")

    # Train XGBoost Instruction Classifier
    # print("Training XGBoost Classifier")
    # xgb_classifier = XGBInstructionClassifier()
    # xgb_classifier.train(features, labels)

    # Compare the models
    print("\nModel Comparison:")
    print(f"RoBERTa Accuracy: {accuracy_roberta:.4f}")
    # print(f"XGBoost Accuracy: {xgb_classifier.model.score(X_test, y_test):.4f}")

    # You can choose which model to use based on the performance
    # For now, let's use the RoBERTa model for further steps

    # Train Model Recommender
    model_recommender = ModelRecommender()
    model_recommender.train(X_train, y_train)

    # Load models for Model Executor
    # models = load_models(config['models'])
    models = load_models()  # Implement this function to load your models
    model_executor = ModelExecutor(models)

    # Create MoEController
    moe_controller = MoEController(roberta_classifier, model_recommender, model_executor)

    # Test MoEController
    test_instructions = [
        "Write a short story about a magical forest.",
        "Explain the concept of quantum entanglement.",
        "Translate 'Hello, how are you?' to French."
    ]
    
    for test_instruction in test_instructions:
        response, recommended_model_index = moe_controller.process_instruction(test_instruction)
        print(f"Test instruction: {test_instruction}")
        print(f"Recommended model index: {recommended_model_index}")
        print(f"Response: {response}")
        print("---")

    # Update models based on feedback (simulated here)
    actual_model_index = 0  # This would be the actual model used
    feedback = 4.5  # This would be the actual feedback score
    moe_controller.update_models(test_instructions[0], actual_model_index, feedback)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train the MoE model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    args = parser.parse_args()
    
    main(num_epochs=args.epochs, batch_size=args.batch_size)
