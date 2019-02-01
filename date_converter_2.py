# coding: utf-8
from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import numpy as np
import argparse
from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
from nmt_utils import *
import matplotlib.pyplot as plt


# # Parse Command Line
# parser = argparse.ArgumentParser(description='Converts an OMD to GLM and runs it on gridlabd')
# parser.add_argument('file_path', metavar='base', type=str,
#                     help='Path to OMD. Put in quotes.')
# args = parser.parse_args()
# filePath = args.file_path

def DateConvert(DATELIST):
    m = 10000
    dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)
    dataset[:10]
    Tx = 30
    Ty = 10
    X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)

    # - `X`: a processed version of the human readable dates in the training set, where each character is replaced by an index mapped to the character via `human_vocab`. Each date is further padded to $T_x$ values with a special character (< pad >). `X.shape = (m, Tx)`
    # - `Y`: a processed version of the machine readable dates in the training set, where each character is replaced by the index it is mapped to in `machine_vocab`. You should have `Y.shape = (m, Ty)`. 
    # - `Xoh`: one-hot version of `X`, the "1" entry's index is mapped to the character thanks to `human_vocab`. `Xoh.shape = (m, Tx, len(human_vocab))`
    # - `Yoh`: one-hot version of `Y`, the "1" entry's index is mapped to the character thanks to `machine_vocab`. `Yoh.shape = (m, Tx, len(machine_vocab))`. Here, `len(machine_vocab) = 11` since there are 11 characters ('-' as well as 0-9). 

    # Defined shared layers as global variables
    repeator = RepeatVector(Tx)
    concatenator = Concatenate(axis=-1)
    densor = Dense(1, activation = "relu")
    activator = Activation(softmax, name='attention_weights') # We are using a custom softmax(axis = 1) loaded in this notebook
    dotor = Dot(axes = 1)



    def one_step_attention(a, s_prev):
        """
        Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
        "alphas" and the hidden states "a" of the Bi-LSTM.
        
        Arguments:
        a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
        s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)
        
        Returns:
        context -- context vector, input of the next (post-attetion) LSTM cell
        """
        s_prev = repeator(s_prev)
        concat = concatenator([a, s_prev])
        e = densor(concat)
        # Use activator and e to compute the attention weights "alphas" (≈ 1 line)
        alphas = activator(e)
        # Use dotor together with "alphas" and "a" to compute the context vector to be given to the next (post-attention) LSTM-cell (≈ 1 line)
        context = dotor([alphas, a])
        return context   

    n_a = 64
    n_s = 128
    post_activation_LSTM_cell = LSTM(n_s, return_state = True)
    output_layer = Dense(len(machine_vocab), activation=softmax)

    def model(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
        """
        Arguments:
        Tx -- length of the input sequence
        Ty -- length of the output sequence
        n_a -- hidden state size of the Bi-LSTM
        n_s -- hidden state size of the post-attention LSTM
        human_vocab_size -- size of the python dictionary "human_vocab"
        machine_vocab_size -- size of the python dictionary "machine_vocab"

        Returns:
        model -- Keras model instance
        """
        
        # Define the inputs of your model with a shape (Tx,)
        # Define s0 and c0, initial hidden state for the decoder LSTM of shape (n_s,)
        X = Input(shape=(Tx, human_vocab_size))
        s0 = Input(shape=(n_s,), name='s0')
        c0 = Input(shape=(n_s,), name='c0')
        s = s0
        c = c0
        
        # Initialize empty list of outputs
        outputs = []
            
        # Step 1: Define your pre-attention Bi-LSTM. Remember to use return_sequences=True. (≈ 1 line)
        a = Bidirectional(LSTM(n_a, return_sequences=True))(X)
        
        # Step 2: Iterate for Ty steps
        for t in range(Ty):
        
            # Step 2.A: Perform one step of the attention mechanism to get back the context vector at step t (≈ 1 line)
            context = one_step_attention(a, s)
            
            # Step 2.B: Apply the post-attention LSTM cell to the "context" vector.
            # Don't forget to pass: initial_state = [hidden state, cell state] (≈ 1 line)
            s, _, c = post_activation_LSTM_cell(context, initial_state = [s, c])
            
            # Step 2.C: Apply Dense layer to the hidden state output of the post-attention LSTM (≈ 1 line)
            out = output_layer(s)
            
            # Step 2.D: Append "out" to the "outputs" list (≈ 1 line)
            outputs.append(out)
        
        # Step 3: Create model instance taking three inputs and returning the list of outputs. (≈ 1 line)
        model = Model(inputs = [X, s0, c0], outputs = outputs)    
        return model


    model = model(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))

    # model.summary()

    out = model.compile(optimizer=Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01),
                        metrics=['accuracy'],
                        loss='categorical_crossentropy')


    s0 = np.zeros((m, n_s))
    c0 = np.zeros((m, n_s))
    outputs = list(Yoh.swapaxes(0,1))


    model.fit([Xoh, s0, c0], outputs, epochs=10, batch_size=100)


    # model.save('date_recognizer_model.h5')

    # model.save_weights('date_recognizer_weights.h5')


    # EXAMPLES = ['3 May 1979', '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007', 'Saturday May 9 2018', 'March 3 2001', 'March 3rd 2001', '1 March 2001']
    # for example in EXAMPLES:
    outputDateList =[]
    for example in DATELIST:
        source = string_to_int(example, Tx, human_vocab)
        source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source)), ndmin=3)
        prediction = model.predict([source, s0, c0])
        prediction = np.argmax(prediction, axis = -1)
        output = [inv_machine_vocab[int(i)] for i in prediction]
        
        print("source:", example)
        print("output:", ''.join(output))
        output = ''.join(output)
        outputDateList.append(output)

    return outputDateList

if __name__ == '__main__':
    DateConvert(DATELIST)






