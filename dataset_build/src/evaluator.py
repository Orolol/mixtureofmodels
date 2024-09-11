from transformers import AutoModelForCausalLM, AutoTokenizer
from base_model import BaseModel

class Evaluator(BaseModel):
    def load_model(self, path):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        return AutoModelForCausalLM.from_pretrained(path)

    def generate(self, instruction):
        inputs = self.tokenizer(instruction, return_tensors="pt")
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=self.parameters.get('max_length', 128),
            temperature=self.parameters.get('temperature', 0.2),
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def evaluate(self, response):
        prompt = f"Evaluate the following response: {response}\nScore (0-10):"
        evaluation = self.generate(prompt)
        try:
            score = float(evaluation.split()[-1])
            return min(max(score, 0), 10)  # Ensure score is between 0 and 10
        except ValueError:
            return 0  # Default score if parsing fails
