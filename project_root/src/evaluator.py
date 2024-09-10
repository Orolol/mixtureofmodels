from transformers import AutoModelForCausalLM, AutoTokenizer

class Evaluator:
    def __init__(self, name, path, parameters):
        self.name = name
        self.model = AutoModelForCausalLM.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.parameters = parameters

    def evaluate(self, response):
        prompt = f"Evaluate the following response: {response}\nScore (0-10):"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=self.parameters.get('max_length', 128),
            temperature=self.parameters.get('temperature', 0.2),
        )
        evaluation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract the numerical score from the evaluation
        try:
            score = float(evaluation.split()[-1])
            return min(max(score, 0), 10)  # Ensure score is between 0 and 10
        except ValueError:
            return 0  # Default score if parsing fails
