# Use Q-Learning for solving the FrozenLake problem
from ailearn.RL.TabularRL import QLearning
from ailearn.RL.Environment import FrozenLake

env = FrozenLake(4)
agent = QLearning(env.n_actions, env.n_states)
MAX_EPISODES = 200

for episode in range(MAX_EPISODES):
    s = env.reset()
    total_reward = 0
    while True:
        a = agent.choose_action(s)
        s_, r, done, _ = env.step(a)
        agent.learn(s, a, r, s_, done)
        s = s_
        total_reward = total_reward + r
        if done:
            break
    print('episode:', episode, 'total_reward:', total_reward)
