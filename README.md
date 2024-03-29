# Simple AI Model - Language Training & Generate text

This repository contains code for training a language model using TensorFlow. The model is built using a sequential architecture with an embedding layer, LSTM layer, and a dense layer with softmax activation. The training process involves tokenizing input text, creating input sequences and labels, and utilizing early stopping for optimization. Additionally, the repository includes functionality to save the trained model and the associated tokenizer. This code can serve as a starting point for language modeling tasks and natural language processing projects.

The part of the "Generate text" code leverages a trained language model to generate text based on a given input. The code loads a pre-trained model and tokenizer, allowing users to interactively generate text by providing an initial input. The generate_text function takes care of predicting the next words in the sequence and iteratively expands the generated text. This module provides a simple yet powerful interface for utilizing the trained language model for creative text generation tasks.

Usage:
Step 1 ( for Google Colab )
<code>!pip install tensorflow</code>

Step 2 
<code>python train.py</code>

Step 3
<code>python generate.py</code>
