from llama_cpp import Llama
from base_model import BaseModel

class ModelWrapper(BaseModel):
    def load_model(self, path):
        return Llama.from_pretrained(
            repo_id=self.name,
            filename=path,
            verbose=True,
            n_ctx=512,
        )

    def generate(self, instruction):
        outputs = self.model.generate(instruction, **self.parameters)
        return self.model.tokenizer.decode(outputs[0], skip_special_tokens=True)

def load_models(model_configs):
    return [ModelWrapper(config['name'], config['path'], config['parameters']) 
            for config in model_configs]
