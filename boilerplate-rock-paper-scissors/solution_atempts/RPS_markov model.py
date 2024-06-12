import random
import tensorflow_probability
import tensorflow as tf
tf.compat.v1.disable_eager_execution()


def player(prev_play, opponent_history=[]) -> str:
    if prev_play != '':
        return random.choice(['R', 'P', 'S'])
    
    #print(f"# game: {len(opponent_history)}")

    # based on the previous play, the key of the dict, count how many times the enemy play R, P or S, the counting is in the internal dict
    occurrences_count: dict[str: dict] = {'R': {'R': 0, 'P': 0, 'S': 0},
                                          'P': {'R': 0, 'P': 0, 'S': 0},
                                          'S': {'R': 0, 'P': 0, 'S': 0}}

    if len(opponent_history) > 1:
        for ind in range(1, len(opponent_history)):
            prev_play = opponent_history[ind - 1]
            current_play = opponent_history[ind]
            if prev_play in occurrences_count and current_play in occurrences_count[prev_play]:
                occurrences_count[prev_play][current_play] += 1

    if len(opponent_history) == 0:
        return random.choice(['R', 'P', 'S'])

    enemy_last_move = opponent_history[-1]

    list_plays = ['R', 'P', 'S']
    transitions_prob = []

    for prev_play in list_plays:
        num_transitions = sum(occurrences_count[prev_play].values())
        if num_transitions == 0:
            transitions_prob.append([1 / 3, 1 / 3, 1 / 3])
        else:
            chance_transition2R = occurrences_count[prev_play]['R'] / num_transitions
            chance_transition2P = occurrences_count[prev_play]['P'] / num_transitions
            chance_transition2S = occurrences_count[prev_play]['S'] / num_transitions
            transitions_prob.append([chance_transition2R, chance_transition2P, chance_transition2S])

    # Ensure transition probabilities are numpy arrays
    transitions_prob = tf.convert_to_tensor(transitions_prob, dtype=tf.float32)

    tfd = tensorflow_probability.distributions
    # Define initial probabilities
    initial_distribution = tfd.Categorical(probs=[1 / 3, 1 / 3, 1 / 3])

    # Define the transition probabilities
    transition_distribution = tfd.Categorical(probs=transitions_prob)

    observation_distribution = tfd.Categorical(probs=[[1/3, 1/3, 1/3],
                                                      [1/3, 1/3, 1/3],
                                                      [1/3, 1/3, 1/3]])

    model = tfd.HiddenMarkovModel(
        initial_distribution=initial_distribution,
        transition_distribution=transition_distribution,
        observation_distribution=observation_distribution,
        num_steps=1
    )

    samples = model.sample(sample_shape=1)

    counter_move = {'R': 'P', 'P': 'S', 'S': 'R'}
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        next_move_index = sess.run(samples)[0][0]
        predicted_move = list_plays[next_move_index]
        return counter_move[predicted_move]
