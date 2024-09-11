import yaml
import uuid
import pandas as pd
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
                score = evaluator.evaluate(instruction, response)
                evaluation_id = str(uuid.uuid4())
                dataset_builder.add_evaluation(evaluation_id, response_id, score)

    # Sauvegarder les datasets
    dataset_builder.save_datasets("dataset_build/output/dataset")

    # Afficher les statistiques
    print_statistics(dataset_builder)

def print_statistics(dataset_builder):
    print("\n--- Statistiques de l'exécution ---")

    # Nombre total de questions
    total_questions = len(dataset_builder.dataset_df)
    print(f"Nombre total de questions: {total_questions}")

    # Nombre de modèles
    num_models = len(dataset_builder.models_df)
    print(f"Nombre de modèles: {num_models}")

    # Statistiques par modèle
    merged_df = pd.merge(dataset_builder.responses_df, dataset_builder.evaluations_df, on='response_id')
    merged_df = pd.merge(merged_df, dataset_builder.models_df, on='model_id')

    for _, model in dataset_builder.models_df.iterrows():
        model_id = model['model_id']
        model_name = model['model_name']
        model_data = merged_df[merged_df['model_id'] == model_id]
        
        mean_score = model_data['score'].mean()
        num_responses = len(model_data)
        
        print(f"\nModèle: {model_name}")
        print(f"  - Nombre de réponses: {num_responses}")
        print(f"  - Score moyen: {mean_score:.2f}")

    # Statistiques globales
    total_responses = len(dataset_builder.responses_df)
    total_evaluations = len(dataset_builder.evaluations_df)
    overall_mean_score = dataset_builder.evaluations_df['score'].mean()

    print(f"\nStatistiques globales:")
    print(f"  - Nombre total de réponses: {total_responses}")
    print(f"  - Nombre total d'évaluations: {total_evaluations}")
    print(f"  - Score moyen global: {overall_mean_score:.2f}")

if __name__ == "__main__":
    main()
