import random

# Initialize Q-table
Q = {}
actions = ['R', 'P', 'S']

def get_q(state, action):
    return Q.get((state, action), 0.0)

def choose_action(state):
    if random.uniform(0, 1) < 0.1:  # Exploration
        return random.choice(actions)
    else:  # Exploitation
        q_values = [get_q(state, a) for a in actions]
        max_q = max(q_values)
        count_max = q_values.count(max_q)
        if count_max > 1:
            best_actions = [actions[i] for i in range(len(actions)) if q_values[i] == max_q]
            return random.choice(best_actions)
        else:
            return actions[q_values.index(max_q)]

def learn(state, action, reward, next_state):
    old_q = get_q(state, action)
    max_q_next = max([get_q(next_state, a) for a in actions])
    Q[(state, action)] = old_q + 0.1 * (reward + 0.9 * max_q_next - old_q)

def get_reward(player_move, opponent_move):
    if player_move == opponent_move:
        return 0  # Draw
    elif (player_move == 'R' and opponent_move == 'S') or (player_move == 'P' and opponent_move == 'R') or (player_move == 'S' and opponent_move == 'P'):
        return 1  # Win
    else:
        return -1  # Loss

# Main loop to train the model
def player(prev_play, opponent_history=[]):
    state = ''.join(opponent_history[-2:]) if len(opponent_history) >= 2 else ''
    action = choose_action(state)
    if prev_play:
        opponent_history.append(prev_play)
        reward = get_reward(action, prev_play)
        next_state = ''.join(opponent_history[-2:])
        learn(state, action, reward, next_state)
    return action
