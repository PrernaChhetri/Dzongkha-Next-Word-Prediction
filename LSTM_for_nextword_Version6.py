# Import libraries
import numpy as np
import re
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense  
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import json

# load dataset
file_path = 'Dataset100k.xlsx'
# Load the Excel file into a DataFrame
df = pd.read_excel(file_path, sheet_name='Sheet1', usecols=['Dzongkha'])

# Extract the text data from a specific column (e.g., 'Dzongkha')
text = df['Dzongkha'].to_string(index=False)
print(text)

# Preprocessing the text
text = re.sub(r'[a-zA-Z0-9]', '', text)
text = text.replace('#', '')
text = text.replace('?', '')
text = text.replace('།', '')
#Removing tsa after the word
text = text.replace(' ', '$')
text = text.replace('་$', '$་')
text = text.replace('$$', '')
text = text.replace('$', ' ')
text = text.replace('་ ', '')
text = text.replace(' ་', ' ')
print(text)

# Tokenize the words
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1
lines = text.split('\n')

# Tokenize and form n-grams for each line separately
input_sequences = []
for line in lines:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)
print(n_gram_sequence)
print(input_sequences)  

# Save the Tokenizer 
tokenizer_file_path = 'Model/LSTM/100k_v6_LSTM_tokenizer.json' 
tokenizer_json = tokenizer.to_json()

import os
# Ensure the directory exists
os.makedirs(os.path.dirname(tokenizer_file_path), exist_ok=True)

with open(tokenizer_file_path, 'w', encoding='utf-8') as f:
    f.write(tokenizer_json)

#input sequences are padded to have same length
max_sequence_len = max([len(seq) for seq in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
print(input_sequences) 
print(max_sequence_len) 

# Split sequences into input and output
X = input_sequences[:, :-1]
y = input_sequences[:, -1]

# Before creating the model, let's verify the shape of our input sequences
print("Max sequence length:", max_sequence_len)
print("Shape of X:", X.shape)

def create_model(total_words, input_length):
    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=max_sequence_len - 1))
    model.add(LSTM(128))
    model.add(Dense(total_words, activation = 'softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

#split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#create a model
model = create_model(total_words, input_length=X_train.shape)

#train the model
history = model.fit(
    X_train, 
    y_train,
    epochs = 100, 
    batch_size = 64,
    verbose = 1,
    validation_data = (X_test, y_test)
)

#Plotting the training and validation accuracy and loss

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
 
plt.tight_layout()
plt.show()
plt.savefig('LSTM_v6_training_validation_metrics.png')

#testing the model
seed_text = input("Input the seed text: ")
next_words_text = 1
next_words = int(next_words_text)

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted_probs = model.predict(token_list)

    # Get the top N predictions
    top_n = 10  # You can adjust the number of top predictions as needed
    top_indices = predicted_probs[0].argsort()[-top_n:][::-1]

    # Create a list of words corresponding to the top indices
    top_words = [word for word, index in tokenizer.word_index.items() if index in top_indices]

    # Output the top predicted words
    print("Top Predicted Words:", top_words)

#save the model
model.save('Model/LSTM/100k_v6_LSTM.h5')


model.save('Model/LSTM/100k_v6_LSTM_savedmodel')


# Save the Tokenizer 
import pickle

tokenizer_file_path = 'Model/LSTM/100k_v6_LSTM_tokenizer.pickle' 
with open(tokenizer_file_path, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

