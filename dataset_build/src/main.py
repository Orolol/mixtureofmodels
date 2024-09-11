import yaml
import uuid
from dataset_build.src.data_loader import load_datasets
from dataset_build.src.model_loader import load_models
from dataset_build.src.evaluator import Evaluator
from dataset_build.src.dataset_builder import DatasetBuilder
import torch

def main():
    print("Check if CUDA is available")
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))

    # Charger la configuration
    with open('dataset_build/config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Charger les datasets
    datasets = load_datasets(config['datasets'])

    # Initialiser le constructeur de dataset
    dataset_builder = DatasetBuilder()

    # Charger les modèles avec leurs paramètres spécifiques
    model_configs = config['models']

    # Pour chaque modèle
    for i, model_config in enumerate(model_configs):
        print(f"Processing model: {model_config['name']}")
        
        # Charger le modèle
        model = load_models([model_config])[0]
        
        # Ajouter le modèle au dataset_builder
        dataset_builder.add_model(i, model.name, model.parameters)

        # Pour chaque dataset chargé
        for dataset_name, dataset in datasets:
            # Ajouter les entrées du dataset chargé au dataset final (si ce n'est pas déjà fait)
            if i == 0:
                dataset_builder.add_entries_from_loaded_dataset(dataset)

            # Pour chaque instruction dans le dataset
            for item in dataset:
                question_id = item['id']
                instruction = item['instruction']
                
                # Générer la réponse
                response = model.generate(instruction)
                response_id = str(uuid.uuid4())
                dataset_builder.add_response(response_id, question_id, i, response)

        # Décharger le modèle pour libérer la mémoire
        del model
        torch.cuda.empty_cache()

    # Charger l'évaluateur
    print("Loading evaluator")
    evaluator = Evaluator(config['evaluator_model']['name'],
                          config['evaluator_model']['path'],
                          config['evaluator_model']['parameters'])

    # Évaluer toutes les réponses
    for _, row in dataset_builder.responses_df.iterrows():
        question_id = row['question_id']
        response_id = row['response_id']
        response = row['response']
        
        # Récupérer l'instruction correspondante
        instruction = dataset_builder.dataset_df.loc[dataset_builder.dataset_df['question_id'] == question_id, 'instruction'].iloc[0]
        
        # Récupérer la réponse existante si elle existe
        existing_response = dataset_builder.dataset_df.loc[dataset_builder.dataset_df['question_id'] == question_id, 'response'].iloc[0] if 'response' in dataset_builder.dataset_df.columns else ''

        # Évaluer la réponse
        score = evaluator.evaluate(instruction, response, existing_response)
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
