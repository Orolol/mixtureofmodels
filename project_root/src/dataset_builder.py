import pandas as pd

class DatasetBuilder:
    def __init__(self):
        self.data = []

    def add_entry(self, instruction, responses, scores):
        entry = {
            'instruction': instruction,
            'responses': responses,
            'scores': scores
        }
        self.data.append(entry)

    def get_dataset(self):
        return pd.DataFrame(self.data)

    def save_dataset(self, path):
        df = self.get_dataset()
        df.to_csv(path, index=False)
