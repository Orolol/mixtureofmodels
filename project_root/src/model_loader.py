from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelWrapper:
    def __init__(self, name, path, parameters):
        self.name = name
        self.model = AutoModelForCausalLM.from_pretrained(name, gguf_file=path)
        self.tokenizer = AutoTokenizer.from_pretrained(name, gguf_file=path)
        self.parameters = parameters

    def generate(self, instruction):
        inputs = self.tokenizer(instruction, return_tensors="pt")
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=self.parameters.get('max_length', 512),
            temperature=self.parameters.get('temperature', 0.7),
            top_p=self.parameters.get('top_p', 0.9),
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

def load_models(model_configs):
    return [ModelWrapper(config['name'], config['path'], config['parameters']) 
            for config in model_configs]
