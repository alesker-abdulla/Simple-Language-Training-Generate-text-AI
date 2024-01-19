import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the pretrained model
pretrained_model = load_model('generated_model.h5')

# Read data from the new dataset file
new_file_path = 'sample_dataset_2.hh'  # Replace with the actual path to your new dataset file

with open(new_file_path, 'r', encoding='utf-8') as new_file:
    new_texts = new_file.readlines()

# Tokenize the new text
new_tokenizer = Tokenizer()
new_tokenizer.fit_on_texts(new_texts)
new_total_words = len(new_tokenizer.word_index) + 1

# Save tokenizer
with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(new_tokenizer, handle)

# Create input sequences and labels
new_input_sequences = []
for line in texts:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        new_input_sequences.append(n_gram_sequence)

new_max_sequence_length = max([len(seq) for seq in new_input_sequences])
print("Max Sequence Length:", new_max_sequence_length)
new_input_sequences = pad_sequences(new_input_sequences, maxlen=new_max_sequence_length, padding='pre')

new_X, new_y = new_input_sequences[:, :-1], new_input_sequences[:, -1]
new_y = tf.keras.utils.to_categorical(new_y, num_classes=new_total_words)

# Fine-tune the existing model on the new dataset
pretrained_model.fit(new_X, new_y, epochs=10, verbose=1)

# Save the updated model with the same name
pretrained_model.save('generated_model.h5')
