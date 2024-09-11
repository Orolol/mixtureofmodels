from data_processor import load_data, preprocess_data, split_data
from instruction_classifier import InstructionClassifier
from model_recommender import ModelRecommender

def main():
    # Load and preprocess data
    data = load_data('../dataset_build/output/dataset_dataset.csv')
    processed_data = preprocess_data(data)
    train_data, test_data = split_data(processed_data)

    # Train Instruction Classifier
    instruction_classifier = InstructionClassifier()
    instruction_classifier.train(train_data['instruction'], train_data['abilities'])

    # Use Instruction Classifier to generate features for Model Recommender
    train_features = instruction_classifier.predict(train_data['instruction'])
    
    # Train Model Recommender
    model_recommender = ModelRecommender()
    model_recommender.train(train_features, train_data['best_model'])

    # Evaluate on test data
    test_features = instruction_classifier.predict(test_data['instruction'])
    predictions = model_recommender.predict(test_features)

    # TODO: Implement evaluation metrics and reporting

if __name__ == "__main__":
    main()
