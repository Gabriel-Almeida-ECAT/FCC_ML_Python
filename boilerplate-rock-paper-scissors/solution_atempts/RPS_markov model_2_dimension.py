import random
import tensorflow_probability
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tfd = tfp.distributions


def player(prev_play, opponent_history=None):
    if len(opponent_history) == 1000:
        opponent_history = []
    
    if prev_play == '':
        return random.choice(['R', 'P', 'S'])

    elif prev_play != '':
        opponent_history.append(prev_play)

    # Initialize occurrences_count for second-order Markov model
    occurrences_count = {'RR': {'R': 0, 'P': 0, 'S': 0},
                         'RP': {'R': 0, 'P': 0, 'S': 0},
                         'RS': {'R': 0, 'P': 0, 'S': 0},
                         'PR': {'R': 0, 'P': 0, 'S': 0},
                         'PP': {'R': 0, 'P': 0, 'S': 0},
                         'PS': {'R': 0, 'P': 0, 'S': 0},
                         'SR': {'R': 0, 'P': 0, 'S': 0},
                         'SP': {'R': 0, 'P': 0, 'S': 0},
                         'SS': {'R': 0, 'P': 0, 'S': 0}}

    if len(opponent_history) > 2:
        for ind in range(2, len(opponent_history)):
            prev_plays = opponent_history[ind - 2] + opponent_history[ind - 1]
            current_play = opponent_history[ind]
            if prev_plays in occurrences_count and current_play in occurrences_count[prev_plays]:
                occurrences_count[prev_plays][current_play] += 1

    if len(opponent_history) < 2:
        return random.choice(['R', 'P', 'S'])

    enemy_last_two_moves = opponent_history[-2] + opponent_history[-1]

    list_plays = ['R', 'P', 'S']
    transitions_prob = []

    for prev_plays in occurrences_count:
        num_transitions = sum(occurrences_count[prev_plays].values())
        if num_transitions == 0:
            transitions_prob.append([1/3, 1/3, 1/3])
        else:
            chance_transition2R = (occurrences_count[prev_plays]['R'] + 1) / (num_transitions + 3)  # Laplace smoothing
            chance_transition2P = (occurrences_count[prev_plays]['P'] + 1) / (num_transitions + 3)  # Laplace smoothing
            chance_transition2S = (occurrences_count[prev_plays]['S'] + 1) / (num_transitions + 3)  # Laplace smoothing
            transitions_prob.append([chance_transition2R, chance_transition2P, chance_transition2S])

    # Ensure transition probabilities are numpy arrays
    transitions_prob = tf.convert_to_tensor(transitions_prob, dtype=tf.float32)

    # Define initial probabilities
    initial_distribution = tfd.Categorical(probs=[1/9] * 9)  # Since there are 9 possible states for second-order

    # Define the transition probabilities
    transition_distribution = tfd.Categorical(probs=transitions_prob)

    # Define the observation probabilities, assuming equal probabilities initially
    observation_distribution = tfd.Categorical(probs=[[1/3, 1/3, 1/3]] * 9)

    model = tfd.HiddenMarkovModel(
        initial_distribution=initial_distribution,
        transition_distribution=transition_distribution,
        observation_distribution=observation_distribution,
        num_steps=1
    )

    # Sample the next state from the model
    samples = model.sample(sample_shape=1)

    counter_move = {'R': 'P', 'P': 'S', 'S': 'R'}
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        next_move_index = sess.run(samples)[0][0]
        predicted_move = list_plays[next_move_index % 3]  # Mapping back to R, P, S
        return counter_move[predicted_move]