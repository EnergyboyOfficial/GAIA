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
    except FileNotFoundError:
        print(f"Tokenizer file not found. Creating a new one.")
        return None
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

def load_best_model(model_path='best_model.h5', tokenizer_path='best_tokenizer.json', accuracy_path='best_accuracy.txt'):
    model = None
    tokenizer = None
    try:
        model = tf.keras.models.load_model(model_path)
        tokenizer = load_tokenizer(tokenizer_path)
        if model is None or tokenizer is None:
            raise FileNotFoundError("Model or tokenizer is None after loading.")
    except (FileNotFoundError, OSError) as e:
        print(f"Error loading existing model or tokenizer: {e}")
        print("Creating a new one.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        print("Creating a new one.")
    return model, tokenizer

def save_best_model(model, tokenizer, accuracy, model_path='best_model.h5', tokenizer_path='best_tokenizer.json', accuracy_path='best_accuracy.txt'):
    model.save(model_path)
    if tokenizer:
        with open(tokenizer_path, 'w') as f:
            f.write(json.dumps({'word_index': tokenizer.word_index}))
    with open(accuracy_path, 'w') as f:
        f.write(str(accuracy))

def print_gaia_ascii():
    print("""
        GGGGGGGGGGGGG               AAA               IIIIIIIIII               AAA
     GGG::::::::::::G              A:::A              I::::::::I              A:::A
   GG:::::::::::::::G             A:::::A             I::::::::I             A:::::A
  G:::::GGGGGGGG::::G            A:::::::A            II::::::II            A:::::::A
 G:::::G       GGGGGG           A:::::::::A             I::::I             A:::::::::A
G:::::G                        A:::::A:::::A            I::::I            A:::::A:::::A
G:::::G                       A:::::A A:::::A           I::::I           A:::::A A:::::A
G:::::G    GGGGGGGGGG        A:::::A   A:::::A          I::::I          A:::::A   A:::::A        
G:::::G    G::::::::G       A:::::A     A:::::A         I::::I         A:::::A     A:::::A
G:::::G    GGGGG::::G      A:::::AAAAAAAAA:::::A        I::::I        A:::::AAAAAAAAA:::::A
G:::::G        G::::G     A:::::::::::::::::::::A       I::::I       A:::::::::::::::::::::A
 G:::::G       G::::G    A:::::AAAAAAAAAAAAA:::::A      I::::I      A:::::AAAAAAAAAAAAA:::::A
  G:::::GGGGGGGG::::G   A:::::A             A:::::A   II::::::II   A:::::A             A:::::A
   GG:::::::::::::::G  A:::::A               A:::::A  I::::::::I  A:::::A               A:::::A
     GGG::::::GGG:::G A:::::A                 A:::::A I::::::::I A:::::A                 A:::::A
        GGGGGG   GGGGAAAAAAA                   AAAAAAAIIIIIIIIIIAAAAAAA                   AAAAAAA
    """)

def main():
    # Print GAIA ASCII art
    print_gaia_ascii()
    
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
    
    # Load or create tokenizer
    tokenizer_path = 'tokenizer.json'
    tokenizer = load_tokenizer(tokenizer_path)
    if tokenizer is None:
        tokenizer = create_tokenizer(conversations)
        save_tokenizer(tokenizer, tokenizer_path)

    vocab_size = len(tokenizer.word_index) + 1
    print(f"Vocabulary size: {vocab_size}")

    # Prepare data
    max_sequence_length = 10  # Set a value based on the typical number of words in a conversation
    print(f"Max sequence length: {max_sequence_length}\n")
    
    input_sequences = []
    for line in conversations:
        token_list = tokenizer.texts_to_sequences([line])[0]
        input_sequences.append(token_list)

    # Print input sequence information
    print("Number of input sequences:", len(input_sequences))
    if input_sequences:
        print("First input sequence:", input_sequences[0])

    if not input_sequences:
        print("Error: No input sequences found.")
        return
    
    input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')
    X, y = input_sequences[:, :-1], input_sequences[:, -1]
    y = tf.keras.utils.to_categorical(y, num_classes=vocab_size)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load or create model
    model_path = 'best_model.h5'
    model, tokenizer = load_best_model(model_path=model_path, tokenizer_path=tokenizer_path)

    if model is None or tokenizer is None:
        model = create_model(vocab_size, max_sequence_length)

    # Training parameters
    training_attempts = 1
    max_accuracy = 0.0  # Initialize to a lower value
    epochs = 50  # Set the number of epochs for each training attempt

    while True:
        try:
            print(f"\nTraining attempt {training_attempts} - Epochs: {epochs} - ETA: Calculating...")
            history = model.fit(X_train, y_train, epochs=epochs, verbose=1)

            # Evaluate the model
            _, current_accuracy = model.evaluate(X_test, y_test, verbose=0)
            print(f"Training attempt {training_attempts} - Epochs: {epochs} - Accuracy: {current_accuracy:.4f}")

            if current_accuracy > max_accuracy:
                print(f"Desired accuracy of {max_accuracy * 100}% achieved! Stopping training.")
                save_best_model(model, tokenizer, current_accuracy, model_path=model_path, tokenizer_path=tokenizer_path)
                break

            # Load the best model for the next attempt
            model, tokenizer = load_best_model(model_path=model_path, tokenizer_path=tokenizer_path)
            if model is None or tokenizer is None:
                model = create_model(vocab_size, max_sequence_length)

            training_attempts += 1

        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Exiting.")
            break

    print("Auto-diagnostics: Training completed.")

if __name__ == "__main__":
    main()
