from llama_cpp import Llama
from dataset_build.src.base_model import BaseModel

class Evaluator(BaseModel):
    def load_model(self, path, name):
        return Llama.from_pretrained(
            repo_id=name,
            filename=path,
            verbose=True,
            n_ctx=512,
        )

    def generate(self, instruction):
        outputs = self.model.generate(instruction, **self.parameters)
        return self.model.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def evaluate(self, response):
        prompt = f"Evaluate the following response: {response}\nScore (0-10):"
        evaluation = self.generate(prompt)
        try:
            score = float(evaluation.split()[-1])
            return min(max(score, 0), 10)  # Ensure score is between 0 and 10
        except ValueError:
            return 0  # Default score if parsing fails
