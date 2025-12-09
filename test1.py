import gymnasium as gym

import numpy as np

import matplotlib.pyplot as plt
environment = gym.make("FrozenLake-v1", is_slippery=False,
                       render_mode='rgb_array')

n_state = environment.observation_space.n
n_action = environment.action_space.n

qtable = np.zeros((environment.observation_space.n, environment.action_space.n))
 
episodes = 1000
# будем учитывать успешность каждого эпизода
success = []
 
for _ in range(episodes):
 
  state = environment.reset()[0]
  terminated = False
  # по умолчанию, будем считать эпизод неудачным
  success.append(0)
 
  while not terminated:
 
    if np.max(qtable[state]) > 0:
      action = np.argmax(qtable[state])
    else:
      action = environment.action_space.sample()
 
    new_state, reward, terminated, *_ = environment.step(action)
 
    qtable[state, action] = reward + np.max(qtable[new_state])
 
    state = new_state
 
    # если же текущий эпизод закончится получением награды,
    if reward:
      # изменим запись о нем с нуля на единицу
      success[-1] = 1
print(qtable)
print(sum(success))
plt.figure(figsize=(10, 4))
plt.bar(range(len(success)), success, width=1.0)
plt.xlabel('Эпизоды')
plt.yticks([0,1])
plt.ylabel('Результат')
plt.show()