class ModelExecutor:
    def __init__(self, models):
        self.models = models

    def execute(self, instruction, model_index):
        if model_index < 0 or model_index >= len(self.models):
            raise ValueError(f"Invalid model index: {model_index}")
        
        model = self.models[model_index]
        return model.generate(instruction)
