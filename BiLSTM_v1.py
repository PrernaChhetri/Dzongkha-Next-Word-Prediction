# Import Libraries
import numpy as np
import pandas as pd
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow.keras.utils as ku
import matplotlib.pyplot as plt
import os

# Load Dataset
file_path = 'Dataset100k.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1', usecols=['Dzongkha'])

# Extract Text Data
text = df['Dzongkha'].to_string(index=False)
print(text)

# Preprocess Text
text = text.replace('#', '')
text = text.replace('?', '')
text = text.replace('།', '')
text = re.sub(r'[a-zA-Z0-9]', '', text)

#Removing the dzongkha period after the word
text = text.replace(' ','$')
text = text.replace('་$', '$་')
text = text.replace('$$','')
text = text.replace('$',' ')
text = text.replace('་ ','')
text = text.replace(' ་',' ')
print(text)

# Data Cleaning
dzo_word = text.split('\n')
all_words = [word for sentence in dzo_word for word in sentence.split()]
print(f"Total number of words in sentences: {len(all_words)}")
print(f"Total number of sentences: {len(dzo_word)}")
print(f"There are {len(set(dzo_word))} unique words")

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(dzo_word)
print(f"No. of unique words in corpus: {len(tokenizer.word_index)}")

# Print Tokenizer's word_index and index for 'ག་'
print("Tokenizer word index:")
print(tokenizer.word_index)
print(f"Index of 'ག་': {tokenizer.word_index.get('ག་', 'Not found')}")

# Generate Sequences
def get_sequence_of_tokens(corpus):
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    return input_sequences, len(tokenizer.word_index) + 1

X, vocab_size = get_sequence_of_tokens(dzo_word)

# Padding Sequences
max_sequence_len = max([len(seq) for seq in X])
input_sequences = np.array(pad_sequences(X, maxlen=max_sequence_len, padding='pre'))

# Categorize Sequences Based on Their Length
more80 = []
less80 = []
for i in X:
    if len(i) > 80:
        more80.append(i)
    else:
        less80.append(i)

print('Length of X is:', len(X))
print('Number of sequences longer than 80 tokens:', len(more80))
print('Number of sequences 80 tokens or shorter:', len(less80))
print('Total of more and less is:', len(more80) + len(less80))
print('The maximum sequence length of the sentence: ', max_sequence_len)

# Split into Predictors and Label
predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
label = ku.to_categorical(label, num_classes=vocab_size)

# Build Model
model = Sequential([
    Embedding(vocab_size, 300, input_length=max_sequence_len-1),
    Bidirectional(LSTM(256, return_sequences=True)),
    Dropout(0.25),
    Bidirectional(LSTM(256, return_sequences=True)),
    Dropout(0.25),
    Bidirectional(LSTM(256)),
    Dropout(0.25),
    Dense(vocab_size//2, activation='relu'),
    Dense(vocab_size, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model Summary
model.summary()

# Callbacks
es = EarlyStopping(monitor='loss', patience=3, verbose=1, mode='min', restore_best_weights=True)
mc = ModelCheckpoint('best_Weights.h5', monitor='loss', verbose=1, save_best_only=True, save_weights_only=True)

# Train Model
history = model.fit(predictors, label, epochs=300, verbose=1, batch_size=128, callbacks=[es, mc])

# Save Tokenizer
tokenizer_file_path = '100k_PW_tokenizer.json'
os.makedirs(os.path.dirname(tokenizer_file_path), exist_ok=True)
with open(tokenizer_file_path, 'w', encoding='utf-8') as f:
    f.write(tokenizer.to_json())

# Plot Training Accuracy and Loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('Model Training Accuracy and Loss')
plt.ylabel('Value')
plt.xlabel('Epoch')
plt.legend(['Accuracy', 'Loss'], loc='upper left')
plt.show()

# Generate Sequence
def generate_seq(model, tokenizer, seq_length, seed_text, n_words=1):
    in_text = seed_text
    for _ in range(n_words):
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        preds = model.predict(encoded)
        pred_word = ''
        for word, index in tokenizer.word_index.items():
            if index == np.argmax(preds):
                pred_word = word
                break
        in_text += ' ' + pred_word
    return in_text

seed_text = "དེ"
generated_text = generate_seq(model, tokenizer, max_sequence_len, seed_text, 10)
print("Seed text:", seed_text)
print("Generated text:", generated_text)

# Save Model
model.save('100K_PW_model.h5')

# Plotting Functions
def plot_graph(history, metric):
    plt.plot(history.history[metric])
    plt.title(f'Model {metric}')
    plt.ylabel(metric)
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

plot_graph(history, 'accuracy')
plot_graph(history, 'loss')

# Model Accuracy and Loss Visualization
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# Enhanced Sequence Generation with Top Predictions
def generate_seq_enhanced(model, tokenizer, seq_length, seed_text, num_generated=10):
    text_result = seed_text
    for _ in range(num_generated):
        encoded = tokenizer.texts_to_sequences([text_result])[0]
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        preds = model.predict(encoded, verbose=0)[0]
        next_index = np.argmax(preds)
        next_word = None
        for word, index in tokenizer.word_index.items():
            if index == next_index:
                next_word = word
                break
        text_result += ' ' + next_word
    return text_result

# Generate and print a sequence using the enhanced function
seed_text = "དེ"
generated_text = generate_seq_enhanced(model, tokenizer, max_sequence_len, seed_text, 10)
print("\nSeed text:", seed_text)
print("Enhanced Generated text:", generated_text)

# Saving the Tokenizer and Model for Deployment
# It's important to have both the model and tokenizer available for future use,
# especially if you plan to deploy the model or use it in another project.

# Save the tokenizer in a pickle file for easy loading
import pickle
tokenizer_pickle_path = '100k_PW_tokenizer.pkl'
with open(tokenizer_pickle_path, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Model has already been saved in H5 format which is suitable for later use
# If you need to save in TensorFlow's SavedModel format for deployment, you can use:
model_save_path = '100K_PW_model_saved_model'
model.save(model_save_path)

print("\nModel and tokenizer have been saved successfully.")
