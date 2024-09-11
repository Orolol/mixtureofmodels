import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

class MoEController:
    def __init__(self, instruction_classifier, model_recommender, model_executor):
        self.instruction_classifier = instruction_classifier
        self.model_recommender = model_recommender
        self.model_executor = model_executor

    def process_instruction(self, instruction):
        try:
            # Step 1: Classify the instruction using the Instruction Classifier
            instruction_features = self.instruction_classifier.predict(instruction)

            # Step 2: Recommend the best model using the Model Recommender
            model_probabilities = self.model_recommender.predict([instruction_features])
            recommended_model_index = np.argmax(model_probabilities[0])

            # Step 3: Execute the recommended model using the Model Executor
            response = self.model_executor.execute(instruction, recommended_model_index)

            logging.info(f"Processed instruction: '{instruction}'. Recommended model index: {recommended_model_index}. Response: {response}")
            return response, recommended_model_index

        except Exception as e:
            logging.error(f"Error processing instruction '{instruction}': {e}")
            return None, None

    def update_models(self, instruction, actual_model_index, feedback):
        try:
            # Step 1: Update the Instruction Classifier
            self.instruction_classifier.update(instruction, actual_model_index, feedback)

            # Step 2: Update the Model Recommender
            instruction_features = self.instruction_classifier.predict(instruction)
            self.model_recommender.update(instruction_features, actual_model_index, feedback)

            logging.info(f"Updated models based on instruction '{instruction}', actual model index {actual_model_index}, and feedback {feedback}")

        except Exception as e:
            logging.error(f"Error updating models for instruction '{instruction}': {e}")
