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
        response = self.model.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": instruction}
            ],
            **self.parameters
        )
        return response['choices'][0]['message']['content']

    def evaluate(self, instruction, response):
        prompt = f"Evaluate the following response to the given instruction. Provide a score between 0 and 10, where 0 is the worst and 10 is the best. Only return the numeric score:\n\nInstruction: {instruction}\n\nResponse: {response}\n\nScore:"
        evaluation = self.generate(prompt)
        try:
            score = float(evaluation.strip())
            return min(max(score, 0), 10)  # Ensure score is between 0 and 10
        except ValueError:
            return 0  # Default score if parsing fails
