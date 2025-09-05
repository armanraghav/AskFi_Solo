import pandas as pd
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
from datasets import Dataset
import numpy as np 

# Load the dataset
try:
    df = pd.read_csv("combined_banking_qa.csv")
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'combined_banking_qa.csv' not found. Please ensure the file is in the correct directory.")
    exit()
except Exception as e:
    print(f"Error loading CSV: {e}")
    print("Please check the CSV file format.")
    exit()

# Drop rows with missing data (good practice)
df.dropna(subset=['question', 'context', 'answer_text'], inplace=True)
print(f"Removed rows with missing data. Remaining samples: {len(df)}")

# Strip whitespaces for uniformity (good practice)
df['question'] = df['question'].str.strip()
df['context'] = df['context'].str.strip()
df['answer_text'] = df['answer_text'].str.strip()

# Add answer start and end positions (character-level)
def add_answer_positions(row):
    context = row['context']
    answer = row['answer_text']
    
    # Find the start of the answer in the context (case-insensitive search)
    start = context.lower().find(answer.lower())
    
    if start == -1:
        print(f"Warning: Answer '{answer}' not found in context '{context}'. Setting start/end to 0.")
        return pd.Series({'start_position': 0, 'end_position': 0})
    else:
        return pd.Series({'start_position': start, 'end_position': start + len(answer)})

print("\nAdding character-level answer positions...")
positions_df = df.apply(add_answer_positions, axis=1)
df = pd.concat([df, positions_df], axis=1)
print("Character-level positions added.")
print(df.head()) 

# Convert DataFrame to Hugging Face Dataset format
dataset = Dataset.from_pandas(df)
print("\nDataset converted to Hugging Face Dataset format.")

# Model and tokenizer setup
model_name = "bert-base-uncased" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
print(f"\nModel and tokenizer '{model_name}' loaded.")

# Tokenization and preprocessing function
def preprocess_function(examples):
    inputs = tokenizer(
        examples['question'],
        examples['context'],
        truncation="only_second",      
        max_length=384,                
        stride=128,                    
        return_overflowing_tokens=True,
        return_offsets_mapping=True,   
        padding="max_length",          
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_mapping = inputs.pop("overflow_to_sample_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_mapping[i]
        
        start_char = examples['start_position'][sample_idx]
        end_char = examples['end_position'][sample_idx]

        sequence_ids = inputs.sequence_ids(i)

        context_start_token = 0
        # Find the actual start token index of the context (sequence_id = 1)
        while context_start_token < len(sequence_ids) and sequence_ids[context_start_token] != 1:
            context_start_token += 1
        
        context_end_token = len(sequence_ids) - 1
        # Find the actual end token index of the context (sequence_id = 1)
        while context_end_token >= 0 and sequence_ids[context_end_token] != 1:
            context_end_token -= 1

        # Handle cases where context might be empty or not found in the chunk
        if context_start_token > context_end_token:
            start_positions.append(0)
            end_positions.append(0)
            continue

        # Check if the answer is completely outside the current context span
        if not (offsets[context_start_token][0] <= start_char and offsets[context_end_token][1] >= end_char):
            start_positions.append(0) 
            end_positions.append(0)
        else:
            token_start_index = context_start_token
            # Move token_start_index to the first token whose character start offset is >= start_char
            while token_start_index <= context_end_token and offsets[token_start_index][0] < start_char:
                token_start_index += 1
            
            token_end_index = context_end_token
            # Move token_end_index to the last token whose character end offset is <= end_char
            while token_end_index >= context_start_token and offsets[token_end_index][1] > end_char:
                token_end_index -= 1
            
            start_positions.append(token_start_index)
            end_positions.append(token_end_index)

    inputs['start_positions'] = start_positions
    inputs['end_positions'] = end_positions
    return inputs

print("\nTokenizing and preprocessing dataset...")
tokenized_dataset = dataset.map(
    preprocess_function, 
    batched=True, 
    remove_columns=dataset.column_names 
)
print("Dataset tokenized.")
print(f"Number of tokenized samples (including overflows): {len(tokenized_dataset)}")

# Training arguments
training_args = TrainingArguments(
    output_dir="./bert_banking_model_v2",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch", 
    report_to="none", 
)
print("\nTraining arguments defined.")

# Create Trainer
trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset)
print("\nTrainer initialized. Starting training...")

# Start training
trainer.train()
print("\nTraining complete.")

# Save the fine-tuned model and tokenizer
model_save_path = "./bert_banking_model_v2"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"\nFine-tuned model and tokenizer saved to '{model_save_path}'.")
