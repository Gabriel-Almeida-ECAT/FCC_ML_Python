import random
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

sequence_size: int = 4

# create the RNN model. Make it global so it isn't nedeed to train every game
model: type[Sequential] = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(sequence_size, 3)))
model.add(Dense(3, activation='softmax')) #output layer, need to be size 3 as there are 3 possibles predictions
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


def encodeMoves(moves):
    encoding: dict[str, list] = {'R': [1, 0, 0], 'P': [0, 1, 0], 'S': [0, 0, 1]}
    return np.array([encoding[move] for move in moves])


def decodeMove(encoded_move):
    return ['R', 'P', 'S'][np.argmax(encoded_move)]


def trainModel(history):
    #x, y = create_dataset(history)
    x: list = [] 
    y: list = []
    
    for i in range(len(history) - sequence_size):
        x.append(history[i:i + sequence_size])
        y.append(history[i + sequence_size])

    x_encoded: type[np.array] = np.array([encodeMoves(seq) for seq in x])
    y_encoded: type[np.array] = encodeMoves(y)

    model.fit(x_encoded, y_encoded, epochs=150, verbose=0)


def player(prev_play, opponent_history = []):
    if len(opponent_history) == 1000:
        opponent_history = []

    game_count: int = len(opponent_history)

    counter_move: dict[str, str] = {'R': 'P', 'P': 'S', 'S': 'R'}
    if prev_play != '' and prev_play != None: # some games the oponent give '' as his move. think its a bug.
        opponent_history.append(prev_play)
        
        # test if model can train yet. Need a minimum amount of enemy plays
        if game_count >= sequence_size+1:
            if game_count < 500 and game_count % 10 == 0:
                print(f'# model trainning: {len(opponent_history)}')
                trainModel(opponent_history)
            
            # slow trainning frequency for better performance
            elif game_count >= 500 and game_count % 20 == 0:
                print(f'# model trainning: {len(opponent_history)}')
                trainModel(opponent_history)

            # predict oponnent next move
            last_moves: list = opponent_history[-sequence_size:]
            encoded_moves: type[np.array] = encodeMoves(last_moves).reshape(1, sequence_size, 3)
            prediction: type[np.array] = model.predict(encoded_moves, verbose=0)

            return counter_move[decodeMove(prediction[0])]

    # for the first games there aren't enough enemy moves to train the model
    return random.choice(['R', 'P', 'S'])