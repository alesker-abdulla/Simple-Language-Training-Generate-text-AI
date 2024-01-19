import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the trained model
single_model = load_model('generated_model.h5')

# Function to generate text using a single model
def generate_text(model, tokenizer, max_sequence_length, input_text, num_words):
    generated_text = input_text
    
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([input_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
        predicted_probabilities = model.predict(token_list, verbose=0)[0]
        
        predicted_index = tf.argmax(predicted_probabilities).numpy()
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
        
        generated_text += " " + output_word
    
    return generated_text

# Sample input
input_text = "Ask something..."

# Tokenize the input text
max_sequence_length = ...  # Use the max sequence length used during training

# Generate information using the single model
generated_info_single = generate_text(single_model, tokenizer, max_sequence_length, input_text, num_words=20)

# Print the generated information
print("Generated Information Single Model:", generated_info_single)
