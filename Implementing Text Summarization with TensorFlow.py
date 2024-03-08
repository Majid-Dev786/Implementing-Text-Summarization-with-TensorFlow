# Importing necessary libraries and modules
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Sample data
DOCUMENTS = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?"
]

SUMMARIES = [
    "<start> First document. <end>",
    "<start> Second document. <end>",
    "<start> Third document. <end>",
    "<start> First document again. <end>"
]

# Class for data preprocessing
class DataPreprocessor:
    def __init__(self, documents, summaries):
        # Initialize a tokenizer with out-of-vocabulary token
        self.tokenizer = Tokenizer(oov_token="<unk>")
        # Fit the tokenizer on both documents and summaries
        self.tokenizer.fit_on_texts(documents + summaries)
        # Get the vocabulary size
        self.vocab_size = len(self.tokenizer.word_index) + 1
        # Initialize maximum document and summary lengths
        self.max_doc_length = 0
        self.max_summary_length = 0
    
    def preprocess(self, documents, summaries):
        # Convert documents and summaries to sequences
        doc_sequences = self.tokenizer.texts_to_sequences(documents)
        summary_sequences = self.tokenizer.texts_to_sequences(summaries)
        # Update maximum document and summary lengths
        self.max_doc_length = max(len(seq) for seq in doc_sequences)
        self.max_summary_length = max(len(seq) for seq in summary_sequences)
        # Pad sequences to the maximum lengths
        padded_doc_sequences = pad_sequences(doc_sequences, maxlen=self.max_doc_length, padding='post')
        padded_summary_sequences = pad_sequences(summary_sequences, maxlen=self.max_summary_length, padding='post')
        return padded_doc_sequences, padded_summary_sequences

# Class for Seq2Seq model
class Seq2SeqModel:
    def __init__(self, vocab_size, max_doc_length, max_summary_length):
        # Initialize vocabulary size and maximum lengths
        self.vocab_size = vocab_size
        self.max_doc_length = max_doc_length
        self.max_summary_length = max_summary_length
        # Build the Seq2Seq model
        self.model = self.build_model()
    
    def build_model(self):
        # Define input layers for document and summary
        input_doc = Input(shape=(self.max_doc_length,))
        input_summary = Input(shape=(self.max_summary_length-1,))
        # Embedding layer for the encoder
        encoder_embedding = Embedding(self.vocab_size, 128)(input_doc)
        # LSTM layer for the encoder
        encoder_outputs, state_h, state_c = LSTM(128, return_state=True)(encoder_embedding)
        encoder_states = [state_h, state_c]
        # Embedding layer for the decoder
        decoder_embedding = Embedding(self.vocab_size, 128)(input_summary)
        # LSTM layer for the decoder with encoder states as initial states
        decoder_lstm = LSTM(128, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
        # Dense layer for output with softmax activation
        decoder_dense = Dense(self.vocab_size, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        # Define the Seq2Seq model
        model = Model([input_doc, input_summary], decoder_outputs)
        # Compile the model with Adam optimizer and categorical crossentropy loss
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def train(self, doc_sequences, summary_sequences):
        # Prepare decoder input data and target data for training
        decoder_input_data = summary_sequences[:, :-1]
        target_data = np.zeros((len(summary_sequences), self.max_summary_length-1, self.vocab_size), dtype='float32')
        for i, seq in enumerate(summary_sequences):
            for j, word_index in enumerate(seq):
                if j > 0:
                    target_data[i, j-1, word_index] = 1.0
        # Train the model
        self.model.fit([doc_sequences, decoder_input_data], target_data, batch_size=64, epochs=10)

# Instantiate the data preprocessor
preprocessor = DataPreprocessor(DOCUMENTS, SUMMARIES)
# Preprocess the data
padded_doc_sequences, padded_summary_sequences = preprocessor.preprocess(DOCUMENTS, SUMMARIES)
# Instantiate the Seq2Seq model
model = Seq2SeqModel(preprocessor.vocab_size, preprocessor.max_doc_length, preprocessor.max_summary_length)
# Train the model
model.train(padded_doc_sequences, padded_summary_sequences)
