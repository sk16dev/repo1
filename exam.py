import os
import fitz  # PyMuPDF
from transformers import T5Tokenizer, T5ForConditionalGeneration


tokenizer = T5Tokenizer.from_pretrained('t5-large')
model = T5ForConditionalGeneration.from_pretrained('t5-large')


def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page_num in range(pdf.page_count):
            page = pdf[page_num]
            text += page.get_text("text")
    return text

def generate_questions(text):
    
    sentences = text.split(".")
    
    questions = [f"What is {sentence}?" for sentence in sentences if len(sentence.strip()) > 0]
    return questions


def answer_questions(contexts):
    answers = []
    for context in contexts:
        input_text = f"question: {context} context: {context}"
        input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        output_ids = model.generate(input_ids, max_length=150, num_beams=5, early_stopping=True)
        answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        answers.append(answer)
    return answers


pdf_path = "llm.pdf"

text = extract_text_from_pdf(pdf_path)

questions = generate_questions(text)

answers = answer_questions(questions)


for i, answer in enumerate(answers):
    print(f"Question {i+1}: {questions[i]}")
    print(f"Answer {i+1}: {answer}")
    print()
