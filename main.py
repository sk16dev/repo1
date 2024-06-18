import os
from dotenv import load_dotenv
from pdf_processor import extract_text_from_pdf
from qa_system import answer_question


load_dotenv()


pdf_path = 'llm.pdf'


text = extract_text_from_pdf(pdf_path)


question = input("ask?")


answer = answer_question(text, question)
print("Question:", question)
print("Answer:", answer)
