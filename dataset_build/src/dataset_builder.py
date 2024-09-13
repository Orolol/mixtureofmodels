import pandas as pd

class DatasetBuilder:
    def __init__(self):
        self.dataset_df = pd.DataFrame(columns=['question_id', 'instruction', 'abilities', 'category', 'language', 'source'])
        self.models_df = pd.DataFrame(columns=['model_id', 'model_name', 'model_parameters'])
        self.responses_df = pd.DataFrame(columns=['response_id', 'question_id', 'model_id', 'response'])
        self.evaluations_df = pd.DataFrame(columns=['evaluation_id', 'response_id', 'score'])

    def add_entry(self, question_id, instruction, abilities, category, language, source):
        new_entry = pd.DataFrame({
            'question_id': [question_id],
            'instruction': [instruction],
            'abilities': [abilities],
            'category': [category],
            'language': [language],
            'source': [source]
        })
        self.dataset_df = pd.concat([self.dataset_df, new_entry], ignore_index=True)

    def add_entries_from_loaded_dataset(self, loaded_dataset):
        new_entries = pd.DataFrame([
            {
                'question_id': item['id'],
                'instruction': item['instruction'],
                'abilities': item['abilities'],
                'category': item['category'],
                'language': item['language'],
                'source': item['source']
            }
            for item in loaded_dataset
        ])
        self.dataset_df = pd.concat([self.dataset_df, new_entries], ignore_index=True)

    def add_model(self, model_id, model_name, model_parameters):
        new_model = pd.DataFrame({
            'model_id': [model_id],
            'model_name': [model_name],
            'model_parameters': [str(model_parameters)]
        })
        self.models_df = pd.concat([self.models_df, new_model], ignore_index=True)

    def add_response(self, response_id, question_id, model_id, response):
        new_response = pd.DataFrame({
            'response_id': [response_id],
            'question_id': [question_id],
            'model_id': [model_id],
            'response': [response]
        })
        self.responses_df = pd.concat([self.responses_df, new_response], ignore_index=True)

    def add_evaluation(self, evaluation_id, response_id, score):
        new_evaluation = pd.DataFrame({
            'evaluation_id': [evaluation_id],
            'response_id': [response_id],
            'score': [score]
        })
        self.evaluations_df = pd.concat([self.evaluations_df, new_evaluation], ignore_index=True)

    def save_datasets(self, base_path):
        self.dataset_df.to_csv(f"{base_path}_dataset.csv", index=False)
        self.models_df.to_csv(f"{base_path}_models.csv", index=False)
        self.responses_df.to_csv(f"{base_path}_responses.csv", index=False)
        self.evaluations_df.to_csv(f"{base_path}_evaluations.csv", index=False)
