from mixture_training.src.data_processor import load_data, preprocess_data, split_data
from mixture_training.src.llm_classifier import LLMClassifier
from mixture_training.src.model_recommender import ModelRecommender
from mixture_training.src.moe_controller import MoEController
from mixture_training.src.model_executor import ModelExecutor
from dataset_build.src.model_loader import load_models
import numpy as np

def main():
    # Load and preprocess data
    processed_data = load_data('dataset_build/output/dataset_instructions.csv')
    features, labels = preprocess_data(processed_data)
    
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
            
    print("Data preprocessed")
    X_train, X_test, y_train, y_test = split_data(features, labels, test_size=0.1)
    print("Data split")

    # Initialize LLM Classifier
    print("Initializing LLM Classifier")
    classifier = LLMClassifier()
    
    # Evaluate the model
    print("Evaluating LLM Classifier")
    y_pred = classifier.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"LLM Classifier Test accuracy: {accuracy:.4f}")

    # Train Model Recommender
    model_recommender = ModelRecommender()
    model_recommender.train(X_train, y_train)

    # Load models for Model Executor
    models = load_models()
    model_executor = ModelExecutor(models)

    # Create MoEController
    moe_controller = MoEController(classifier, model_recommender, model_executor)

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
    main()
