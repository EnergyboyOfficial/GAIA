import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json
import time
import os

def load_conversations(file_path='conversations.txt'):
    # Load conversations from a file
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    conversations = [line.strip() for line in lines]
    return conversations

def create_dataset(conversations):
    # Tokenize the conversations
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(conversations)
    total_words = len(tokenizer.word_index) + 1

    # Create input sequences and labels
    input_sequences = []
    for line in conversations:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    # Pad sequences
    max_sequence_length = max(len(seq) for seq in input_sequences)
    input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

    # Create predictors and labels
    X, y = input_sequences[:, :-1], input_sequences[:, -1]
    y = tf.keras.utils.to_categorical(y, num_classes=total_words)

    return X, y, tokenizer, total_words, max_sequence_length

def build_model(total_words, max_sequence_length):
    # Build the model
    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=max_sequence_length-1))
    model.add(LSTM(100))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def train_model(model, X, y, epochs=50):
    # Train the model
    model.fit(X, y, epochs=epochs, verbose=0)
    return model

def evaluate_model(model, X_test, y_test):
    # Evaluate the model on the test set
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return accuracy

def save_best_model(model, tokenizer, accuracy):
    # Save the best model and tokenizer if the accuracy is higher
    model.save('best_model.h5')
    with open('best_tokenizer.json', 'w') as f:
        f.write(json.dumps({'word_index': tokenizer.word_index}))
    with open('best_accuracy.txt', 'w') as f:
        f.write(str(accuracy))

    print(f"New best accuracy achieved: {accuracy:.4f}. Saving the model.")

def load_best_model():
    # Load the best model and tokenizer
    model = tf.keras.models.load_model('best_model.h5')
    with open('best_tokenizer.json', 'r') as f:
        tokenizer_config = json.load(f)
        tokenizer = Tokenizer()
        tokenizer.word_index = tokenizer_config['word_index']
    return model, tokenizer

def main():
    print("Auto-diagnostics: Initializing...")
    try:
        # Load existing tokenizer
        with open('best_tokenizer.json', 'r') as f:
            tokenizer_config = json.load(f)
        tokenizer = Tokenizer()
        tokenizer.word_index = tokenizer_config['word_index']
        print("Loaded existing tokenizer.")
    except (FileNotFoundError, json.JSONDecodeError):
        print("Error loading existing tokenizer. Creating a new one.")
        tokenizer = None

    # Load conversations
    conversations = load_conversations()
    print(f"Total number of conversations: {len(conversations)}")

    # Create dataset
    X, y, _, total_words, max_sequence_length = create_dataset(conversations)
    print(f"Total number of words: {total_words}")
    print(f"Max sequence length: {max_sequence_length}")

    # Split the data into training and test sets
    split_ratio = 0.8
    split_index = int(split_ratio * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Initialize variables
    best_accuracy = 0.0
    attempt = 1
    max_attempts = 5  # You can adjust this value

    while best_accuracy < 0.8 and attempt <= max_attempts:
        print(f"\nTraining attempt {attempt} - Epochs: 50 - ETA: Calculating...")

        # Build and train the model
        model = build_model(total_words, max_sequence_length)
        if tokenizer:
            model, _ = load_best_model()  # Load the previous best model
        model = train_model(model, X_train, y_train, epochs=50)

        # Evaluate the model on the test set
        accuracy = evaluate_model(model, X_test, y_test)
        print(f"Accuracy: {accuracy:.4f}")

        if accuracy > best_accuracy:
            # Save the best model and tokenizer
            save_best_model(model, tokenizer, accuracy)
            best_accuracy = accuracy
        else:
            # Load the best model if no improvement
            model, tokenizer = load_best_model()
            print(f"Loading the best model with accuracy {best_accuracy:.4f} for the next attempt.")

        # Increment the attempt counter
        attempt += 1

    print("\nAuto-diagnostics: Training completed.")
    print(f"Best accuracy achieved: {best_accuracy:.4f}")

if __name__ == "__main__":
    main()
