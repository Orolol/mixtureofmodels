from datasets import load_dataset
from huggingface_hub import login
import json
from mom_utils.env import load_env_variables, get_huggingface_token

def load_datasets(dataset_configs, max_iter):
    datasets = []
    load_env_variables()
    print("Token", get_huggingface_token())
    login(token=get_huggingface_token())
    for config in dataset_configs:
        if config['path'] == 'BAAI/Infinity-Instruct':
            dataset = load_dataset('BAAI/Infinity-Instruct', '0625', split='train')
        else:
            dataset = load_dataset(config['path'])
        
        # Process the dataset to extract relevant information
        processed_dataset = []
        for i, item in enumerate(dataset):
            if i >= max_iter:  # Process only the first 5 items
                break
            print(f"Processing item {i+1}/{max_iter} : {item['id']}", end="\r", flush=True)
            
            conversation = item['conversations']
            instruction = next(msg['value'] for msg in conversation if msg['from'] == 'human')
            response = next(msg['value'] for msg in conversation if msg['from'] == 'gpt')
            
            abilities = item['label']['ability_en'] if type(item['label']) == dict else ""
            category = item['label']['cate_ability_en'][0] if type(item['label']) == dict and 'cate_ability_en' in item['label'] and len(item['label']['cate_ability_en']) > 0 else ""

            processed_item = {
                'id': item['id'],
                'instruction': instruction,
                'response': response,
                'abilities': abilities,
                'category': category,
                'language': item['langdetect'],
                'source': item['source']
            }
            processed_dataset.append(processed_item)
        
        datasets.append((config['name'], processed_dataset))
    return datasets
