from datasets import load_dataset
import json

def load_datasets(dataset_configs):
    datasets = []
    for config in dataset_configs:
        dataset = load_dataset(config['path'])
        
        # Process the dataset to extract relevant information
        processed_dataset = []
        for item in dataset['train']:  # Assuming 'train' split, adjust if needed
            conversation = json.loads(item['conversations'])
            instruction = next(msg['value'] for msg in conversation if msg['from'] == 'human')
            response = next(msg['value'] for msg in conversation if msg['from'] == 'gpt')
            
            processed_item = {
                'id': item['id'],
                'instruction': instruction,
                'response': response,
                'abilities': item['label']['ability_en'],
                'category': item['label']['cate_ability_en'][0],
                'language': item['langdetect'],
                'source': item['source']
            }
            processed_dataset.append(processed_item)
        
        datasets.append((config['name'], processed_dataset))
    return datasets
