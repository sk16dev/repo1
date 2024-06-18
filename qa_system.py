import os
from transformers import T5Tokenizer, T5ForConditionalGeneration

class OllamaLLM:
    def __init__(self):
        self.api_key = os.getenv("OLLAMA_API_KEY")
        if not self.api_key:
            raise ValueError("OLLAMA_API_KEY environment variable not set")
        self.tokenizer = T5Tokenizer.from_pretrained("t5-large")
        self.model = T5ForConditionalGeneration.from_pretrained("t5-large")

    def _lm_type(self):
        return "specific_type"

    def answer_question(self, text, question):
        input_text = f"question: {question} context: {text}"
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        output_ids = self.model.generate(input_ids, max_length=4096, num_beams=5, early_stopping=True)
        answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return answer

def answer_question(text, question):
    llm = OllamaLLM()
    return llm.answer_question(text, question)
