import environments.create.create_game
import gym

env = gym.make('CreateLevelPush-v0')

env.reset()

done = False
while not done: 
    obs, reward, done, info = env.step(env.action_space.sample())
    env.render('human')