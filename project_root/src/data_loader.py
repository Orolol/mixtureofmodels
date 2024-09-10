from datasets import load_dataset

def load_datasets(dataset_configs):
    datasets = []
    for config in dataset_configs:
        dataset = load_dataset(config['path'])
        datasets.append((config['name'], dataset))
    return datasets
