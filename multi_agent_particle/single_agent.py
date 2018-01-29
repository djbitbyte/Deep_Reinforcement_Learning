import gym

e = gym.make('CartPole-v0')
observation = e.reset()
for i in range(100):
    obs, rew, done, _ = e.step(e.action_space.sample())
    e.render()
    if done:
        e.reset()


