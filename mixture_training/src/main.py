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

def main():
    # Load configuration
    # with open('../dataset_build/config/config.yaml', 'r') as file:
    #     config = yaml.safe_load(file)
    
    # check if the processed data pickle file exists
    if os.path.exists('dataset_build/save/processed_data.pkl'):
        with open('dataset_build/save/processed_data.pkl', 'rb') as f:
            processed_data = pickle.load(f)
    else:
        # Load and preprocess data
        data = load_data('dataset_build/output/dataset_instructions.csv')
        print("Data loaded")
        processed_data = preprocess_data(data)
        
        # Let's save the processed data to a pickle file
        if not os.path.exists('dataset_build/save'):
            os.makedirs('dataset_build/save')
        print("Saving processed data")
        with open('dataset_build/save/processed_data.pkl', 'wb') as f:
            pickle.dump(processed_data, f)
            
    print("Data preprocessed")
    features, labels = processed_data

    X_train, X_test, y_train, y_test = split_data(features, labels=labels)
    print("Data split")

    # Train Neural Network Instruction Classifier
    print("Training Neural Network Classifier")
    nn_classifier = InstructionClassifier(X_train.shape[1], hidden_sizes=[512, 256, 128], num_classes=len(np.unique(y_train)))
    nn_classifier.train(X_train, y_train, num_epochs=200, batch_size=64)
    
    # Evaluate the Neural Network model
    y_pred_nn = nn_classifier.predict(X_test)
    accuracy_nn = np.mean(y_pred_nn.argmax(axis=1) == y_test)
    print(f"Neural Network Test accuracy: {accuracy_nn:.4f}")

    # Train XGBoost Instruction Classifier
    print("Training XGBoost Classifier")
    xgb_classifier = XGBInstructionClassifier()
    xgb_classifier.train(features, labels)

    # Compare the models
    print("\nModel Comparison:")
    print(f"Neural Network Accuracy: {accuracy_nn:.4f}")
    print(f"XGBoost Accuracy: {xgb_classifier.model.score(X_test, y_test):.4f}")

    # You can choose which model to use based on the performance
    # For now, let's use the XGBoost model for further steps

    # Train Model Recommender
    # model_recommender = ModelRecommender()
    # model_recommender.train(train_features, train_data['best_model'])

    # # Load models for Model Executor
    # models = load_models(config['models'])
    # model_executor = ModelExecutor(models)

    # # Create MoEController
    # moe_controller = MoEController(xgb_classifier, model_recommender, model_executor)

    # # Test MoEController
    # test_instruction = "Write a short story about a magical forest."
    # response, recommended_model_index = moe_controller.process_instruction(test_instruction)
    # print(f"Test instruction: {test_instruction}")
    # print(f"Recommended model index: {recommended_model_index}")
    # print(f"Response: {response}")

    # # Update models based on feedback (simulated here)
    # actual_model_index = 0  # This would be the actual model used
    # feedback = 4.5  # This would be the actual feedback score
    # moe_controller.update_models(test_instruction, actual_model_index, feedback)

if __name__ == "__main__":
    main()
