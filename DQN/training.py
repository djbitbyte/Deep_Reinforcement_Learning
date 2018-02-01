from maze_env import Maze
from DQN_brain import DeepQNetwork


def run_maze():
    step = 0
    for episode in range(2):
        observation = env.reset()

        while True:
            env.render()
            action = DQN.choose_action(observation)
            observation_, reward, done = env.step(action)
            DQN.store_transition(observation, action, reward, observation_)

            # store memory
            if (step > 200) and (step % 5 == 0):
                DQN.learn()

            if done:
                break
            step += 1

    print('game over')
    env.destroy()


if __name__ == "__main__":
    env = Maze()
    DQN = DeepQNetwork(env.n_actions, env.n_features,
                       learning_rate=0.01,
                       reward_decay=0.9,
                       e_greedy=0.9,
                       replace_target_iter=200,
                       memory_size=2000,
                       output_graph=False
                       )
    env.after(100, run_maze)
    env.mainloop()
    DQN.plot_cost()
    # print(DQN.n_features)





