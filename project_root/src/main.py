import yaml
from data_loader import load_datasets
from model_loader import load_models
from evaluator import Evaluator
from dataset_builder import DatasetBuilder

def main():
    # Charger la configuration
    with open('../config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Charger les datasets
    datasets = load_datasets(config['datasets'])

    # Charger les modèles avec leurs paramètres spécifiques
    models = load_models(config['models'])

    # Initialiser l'évaluateur avec ses paramètres spécifiques
    evaluator = Evaluator(config['evaluator_model']['name'],
                          config['evaluator_model']['path'],
                          config['evaluator_model']['parameters'])

    # Initialiser le constructeur de dataset
    dataset_builder = DatasetBuilder()

    # Pour chaque instruction dans les datasets
    for dataset_name, dataset in datasets:
        for instruction in dataset['train']:  # Assuming 'train' split, adjust if needed
            # Obtenir les réponses de chaque modèle
            responses = {model.name: model.generate(instruction) for model in models}

            # Évaluer chaque réponse
            scores = {name: evaluator.evaluate(response) for name, response in responses.items()}

            # Ajouter à notre dataset final
            dataset_builder.add_entry(instruction, responses, scores)

    # Sauvegarder le dataset final
    dataset_builder.save_dataset("../output/final_dataset.csv")

if __name__ == "__main__":
    main()
