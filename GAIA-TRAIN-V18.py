import json
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
    return data

def preprocess_data(data):
    # Add any data preprocessing steps here
    return data

def create_tokenizer(conversations):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(conversations)
    return tokenizer

def save_tokenizer(tokenizer, file_path):
    with open(file_path, 'w') as f:
        f.write(json.dumps({'word_index': tokenizer.word_index}))

def load_tokenizer(file_path):
    try:
        with open(file_path, 'r') as f:
            tokenizer_config = json.load(f)
        tokenizer = Tokenizer()
        tokenizer.word_index = tokenizer_config['word_index']
        return tokenizer
    except Exception as e:
        print(f"Error loading existing tokenizer: {e}")
        print("Creating a new one.")
        return None

def create_model(vocab_size, max_sequence_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 50, input_length=max_sequence_length-1))
    model.add(LSTM(100))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def load_best_model():
    try:
        model = tf.keras.models.load_model('best_model.h5')
        tokenizer = load_tokenizer('best_tokenizer.json')
        return model, tokenizer
    except Exception as e:
        print(f"Error loading existing model: {e}")
        print("Creating a new one.")
        return None, None

def save_best_model(model, tokenizer, accuracy):
    model.save('best_model.h5')
    with open('best_tokenizer.json', 'w') as f:
        f.write(json.dumps({'word_index': tokenizer.word_index}))
    with open('best_accuracy.txt', 'w') as f:
        f.write(str(accuracy))

def main():
    # Auto-diagnostics
    print("Auto-diagnostics: Initializing...")
    conversations = preprocess_data(load_data('conversations.txt'))

    total_conversations = len(conversations)
    total_words = len(' '.join(conversations).split())
    
    if total_conversations == 0:
        print("Error: No conversations found.")
        return
    
    max_sequence_length = max(len(conv.split()) for conv in conversations) if total_conversations > 0 else 0
    print(f"Total number of conversations: {total_conversations}")
    print(f"Total number of words: {total_words}")
    print(f"Max sequence length: {max_sequence_length}\n")

    # Load or create tokenizer
    tokenizer_path = 'tokenizer.json'
    tokenizer = load_tokenizer(tokenizer_path)
    if tokenizer is None:
        tokenizer = create_tokenizer(conversations)
        save_tokenizer(tokenizer, tokenizer_path)

    vocab_size = len(tokenizer.word_index) + 1
    print(f"Vocabulary size: {vocab_size}")

    # Prepare data
    input_sequences = []
    for line in conversations:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    if not input_sequences:
        print("Error: No input sequences found.")
        return
    
    max_sequence_length = max(len(seq) for seq in input_sequences)
    input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')
    X, y = input_sequences[:, :-1], input_sequences[:, -1]
    y = tf.keras.utils.to_categorical(y, num_classes=vocab_size)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load or create model
    model, tokenizer = load_best_model()
    if model is None or tokenizer is None:
        model = create_model(vocab_size, max_sequence_length)

    # Training parameters
    training_attempts = 1
    max_accuracy = 0.8  # Set the desired maximum accuracy
    epochs = 50  # Set the number of epochs for each training attempt

    while True:
        print(f"\nTraining attempt {training_attempts} - Epochs: {epochs} - ETA: Calculating...")
        history = model.fit(X_train, y_train, epochs=epochs, verbose=1)

        # Evaluate the model
        _, current_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"Training attempt {training_attempts} - Epochs: {epochs} - Accuracy: {current_accuracy:.4f}")

        if current_accuracy > max_accuracy:
            print(f"Desired accuracy of {max_accuracy * 100}% achieved! Stopping training.")
            save_best_model(model, tokenizer, current_accuracy)
            break

        # Load the best model for the next attempt
        model, tokenizer = load_best_model()
        training_attempts += 1

    print("Auto-diagnostics: Training completed.")

if __name__ == "__main__":
    main()
