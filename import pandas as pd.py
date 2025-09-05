import pandas as pd

# STEP 1 → Load Dataset
df = pd.read_csv("banking_qa.csv")

# STEP 2 → Clean Column Names (removes spaces, makes lowercase)
df.columns = df.columns.str.strip().str.lower()

# STEP 3 → Clean Text Data (removes spaces, makes lowercase)
df['question'] = df['question'].str.strip().str.lower()
df['context'] = df['context'].str.strip().str.lower()
df['answer_text'] = df['answer_text'].str.strip().str.lower()

# OPTIONAL → Check for missing data
print("Null values:\n", df.isnull().sum())

# STEP 4 → Add Answer Positions (start & end positions)
def add_answer_positions(row):
    answer = row['answer_text']
    context = row['context']
    
    # Find where the answer starts in the context
    start_idx = context.find(answer)
    
    # If the answer is NOT found → raise error so you can fix it
    if start_idx == -1:
        raise ValueError(f"Answer '{answer}' not found in context:\n{context}")
        
    # Calculate the end position of the answer
    end_idx = start_idx + len(answer)
    
    return pd.Series({'start_position': start_idx, 'end_position': end_idx})

# Apply function to each row
positions = df.apply(add_answer_positions, axis=1)

# Add the new start and end columns to the dataframe
df = pd.concat([df, positions], axis=1)

# STEP 4.1 → Check final dataframe
print(df[['question', 'context', 'answer_text', 'start_position', 'end_position']].head())
