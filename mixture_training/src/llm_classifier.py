from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LLMClassifier:
    def __init__(self, model_name="meta-llama/Llama-3.2-1B"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

    def classify(self, text):
        prompt = f"""Classify the following instruction into one of these categories, answer with the category number:
1. Communication & Task Management
2. Technical Assistance & Coding Help
3. Creative Content Generation
4. Professional & Specialized Expertise
5. Information Retrieval & General Knowledge

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
