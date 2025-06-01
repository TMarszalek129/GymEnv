import numpy as np

def train_agent(agent, max_steps, diff):
    training_rewards = []
    epsilons = []
    states = [[] for _ in range(8)]
    episode = 0

    while agent.epsilon - agent.epsilon_min > diff:
        episode += 1

        state, info = agent.env.reset()
        for i in range(8):
            states[i].append(state[i])
        state = agent.discretize_state(state)
        total_training_rewards = 0

        for step in range(max_steps):
            action = agent.choose_action(state)
            new_state, reward, terminated, truncated, info = agent.env.step(action)
            for i in range(8):
                states[i].append(state[i])
            new_state = agent.discretize_state(new_state)

            agent.update_Q(state, new_state, action, reward)
            total_training_rewards += reward
            state = new_state

        agent.epsilon = agent.epsilon_min + (agent.epsilon_max - agent.epsilon_min)*np.exp(-agent.epsilon_decay*episode)

        training_rewards.append(total_training_rewards)
        epsilons.append(agent.epsilon)

        print ("Total reward for episode {}: {}, Epsilon: {}".format(episode, sum(training_rewards)/episode, agent.epsilon))

    print("Training score over time: " + str(sum(training_rewards)/episode))

    return training_rewards, epsilons, episode, states

# training_rewards, epsilons, train_episodes, states = train_agent(max_steps, diff=0.001)
