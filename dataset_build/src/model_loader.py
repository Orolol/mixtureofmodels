from llama_cpp import Llama

class ModelWrapper:
    def __init__(self, name, path, parameters):
        self.name = name
        self.model = Llama.from_pretrained(
            repo_id=name,
            filename=path,
            verbose=True,
            n_ctx=512,
        )
        self.parameters = parameters

    def generate(self, instruction):
        outputs = self.model.generate(instruction, **self.parameters)
        return self.model.tokenizer.decode(outputs[0], skip_special_tokens=True)

def load_models(model_configs):
    return [ModelWrapper(config['name'], config['path'], config['parameters']) 
            for config in model_configs]
