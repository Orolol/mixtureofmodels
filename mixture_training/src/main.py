from data_processor import load_data, preprocess_data, split_data
from instruction_classifier import InstructionClassifier
from model_recommender import ModelRecommender
from moe_controller import MoEController
from model_executor import ModelExecutor
from dataset_build.src.model_loader import load_models
import yaml

def main():
    # Load configuration
    with open('../dataset_build/config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Load and preprocess data
    data = load_data('../dataset_build/output/dataset_dataset.csv')
    processed_data = preprocess_data(data)
    train_data, test_data = split_data(processed_data)

    # Train Instruction Classifier
    instruction_classifier = InstructionClassifier()
    instruction_classifier.train(train_data['instruction'], train_data['abilities'])

    # Use Instruction Classifier to generate features for Model Recommender
    train_features = [instruction_classifier.predict(instr) for instr in train_data['instruction']]
    
    # Train Model Recommender
    model_recommender = ModelRecommender()
    model_recommender.train(train_features, train_data['best_model'])

    # Load models for Model Executor
    models = load_models(config['models'])
    model_executor = ModelExecutor(models)

    # Create MoEController
    moe_controller = MoEController(instruction_classifier, model_recommender, model_executor)

    # Test MoEController
    test_instruction = "Write a short story about a magical forest."
    response, recommended_model_index = moe_controller.process_instruction(test_instruction)
    print(f"Test instruction: {test_instruction}")
    print(f"Recommended model index: {recommended_model_index}")
    print(f"Response: {response}")

    # Update models based on feedback (simulated here)
    actual_model_index = 0  # This would be the actual model used
    feedback = 4.5  # This would be the actual feedback score
    moe_controller.update_models(test_instruction, actual_model_index, feedback)

if __name__ == "__main__":
    main()
