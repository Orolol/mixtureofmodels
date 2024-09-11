from llama_cpp import Llama
from dataset_build.src.base_model import BaseModel

class ModelWrapper(BaseModel):
    def load_model(self, path, name):
        return Llama.from_pretrained(
            repo_id=name,
            filename=path,
            verbose=True,
            n_ctx=4096,
            n_gpu_layers=200
        )

    def generate(self, instruction):
        response = self.model.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": instruction}
            ],
            **self.parameters
        )
        return response['choices'][0]['message']['content']

def load_models(model_configs):
    return [ModelWrapper(config['name'], config['path'], config['parameters']) 
            for config in model_configs]
