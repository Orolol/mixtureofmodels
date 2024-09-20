from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LLMClassifier:
    def __init__(self, model_name="microsoft/Phi-3-mini-4k-instruct-gguf"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

    def classify(self, text):
        prompt = f"""Classify the following instruction into one of these categories:
1. Natural Language Processing and Understanding
2. Information Processing and Integration
3. Mathematical Ability
4. Problem Solving and Support
5. Programming and Software Development
6. Data Science and Analytics
7. General Knowledge and Q&A
8. Creative and Artistic Endeavors
9. Language and Culture
10. Business and Finance
11. Analysis and Reasoning
12. Specialized Knowledge
13. Education and Communication
14. Task Management

Instruction: {text}

Category:"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=10, num_return_sequences=1)
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        category = response.split("Category:")[-1].strip()
        
        return category

    def predict(self, texts):
        return [self.classify(text) for text in texts]
