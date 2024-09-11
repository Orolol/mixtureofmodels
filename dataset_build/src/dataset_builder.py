import pandas as pd

class DatasetBuilder:
    def __init__(self):
        self.data = []

    def add_entry(self, id, instruction, response, abilities, category, language, source):
        entry = {
            'id': id,
            'instruction': instruction,
            'response': response,
            'abilities': abilities,
            'category': category,
            'language': language,
            'source': source
        }
        self.data.append(entry)

    def add_entries_from_loaded_dataset(self, loaded_dataset):
        for item in loaded_dataset:
            self.add_entry(
                item['id'],
                item['instruction'],
                item['response'],
                item['abilities'],
                item['category'],
                item['language'],
                item['source']
            )

    def get_dataset(self):
        return pd.DataFrame(self.data)

    def save_dataset(self, path):
        df = self.get_dataset()
        df.to_csv(path, index=False)
