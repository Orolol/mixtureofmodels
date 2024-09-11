import yaml
import uuid
from dataset_build.src.data_loader import load_datasets
from dataset_build.src.model_loader import load_models
from dataset_build.src.evaluator import Evaluator
from dataset_build.src.dataset_builder import DatasetBuilder

def main():
    # Charger la configuration
    with open('dataset_build/config/config.yaml', 'r') as file:
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

    # Ajouter les modèles au dataset_builder
    for i, model in enumerate(models):
        dataset_builder.add_model(i, model.name, model.parameters)

    # Pour chaque dataset chargé
    for dataset_name, dataset in datasets:
        # Ajouter les entrées du dataset chargé au dataset final
        dataset_builder.add_entries_from_loaded_dataset(dataset)

        # Pour chaque instruction dans le dataset
        for item in dataset:
            question_id = item['id']
            instruction = item['instruction']
            
            # Obtenir les réponses de chaque modèle
            for i, model in enumerate(models):
                response = model.generate(instruction)
                response_id = str(uuid.uuid4())
                dataset_builder.add_response(response_id, question_id, i, response)

                # Évaluer la réponse du modèle
                score = evaluator.evaluate(response)
                evaluation_id = str(uuid.uuid4())
                dataset_builder.add_evaluation(evaluation_id, response_id, score)

    # Sauvegarder les datasets
    dataset_builder.save_datasets("dataset_build/output/dataset")

if __name__ == "__main__":
    main()
