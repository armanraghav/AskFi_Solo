from bs4 import BeautifulSoup
import pandas as pd

html_file = "FAQ Personal Loan - Faq's.html"

with open(html_file, 'r', encoding='utf-8') as file:
    soup = BeautifulSoup(file, 'lxml')

questions = []
answers = []

for faq_block in soup.find_all(['h2', 'h3', 'strong', 'b']):
    question_text = faq_block.get_text(strip=True)
    
    answer = faq_block.find_next('p')
    
    if answer:
        answer_text = answer.get_text(strip=True)
        questions.append(question_text)
        answers.append(answer_text)

df = pd.DataFrame({'question': questions, 'answer': answers})

df['context'] = df['answer']
df['answer_start'] = 0
df['answer_text'] = df['answer']

df[['context', 'question', 'answer_start', 'answer_text']].to_csv("personal_loan_faq.csv", index=False)

print("✅ Extraction complete → File saved as: personal_loan_faq.csv")
