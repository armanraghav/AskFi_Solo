import pandas as pd

df1 = pd.read_csv("banking_qa.csv")          # old
df2 = pd.read_csv("personal_loan_faq.csv")   # new
df_combined = pd.concat([df1, df2], ignore_index=True)
df_combined.to_csv("combined_banking_qa.csv", index=False)
