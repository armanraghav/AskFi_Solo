from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

model_dir = r"C:\Users\Armaan Raghav\Desktop\Internship\Project_Internship\bert_banking_model_v2"

model = AutoModelForQuestionAnswering.from_pretrained(model_dir, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

context = """
A savings account is a deposit account held at a bank that earns interest. Savings accounts typically pay interest on your deposits.
"""
question = "What is a savings account?"

result = qa_pipeline(question=question, context=context)
print("Answer:", result['answer'])

#pdfs-,Convert into chunks/preprocessing,faiss(vector),Semi automate,
#can set limits
#pdfs summary button in ui 
#