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

    # Pour chaque dataset chargé
    for dataset_name, dataset in datasets:
        # Ajouter les entrées du dataset chargé au dataset final
        dataset_builder.add_entries_from_loaded_dataset(dataset)

        # Pour chaque instruction dans le dataset
        for item in dataset:
            instruction = item['instruction']
            
            # Obtenir les réponses de chaque modèle
            model_responses = {model.name: model.generate(instruction) for model in models}

            # Évaluer chaque réponse de modèle
            model_scores = {name: evaluator.evaluate(response) for name, response in model_responses.items()}

            # TODO: Decide how to handle and store model responses and scores
            # You might want to add these to the dataset_builder or store them separately

    # Sauvegarder le dataset final
    dataset_builder.save_dataset("../output/final_dataset.csv")

if __name__ == "__main__":
    main()
