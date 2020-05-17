import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

for name in ['dqn_gamma_boxing_0.99.log', 'dqn_gamma_DDQN_boxing_0.99.log']:
    epochs = []
    rewards = []
    with open(name, 'r') as f:
        for line in f:
            _, epoch, _, _, _, _, _, _, reward, _, _, _, _, _, _ = line.split()
            epoch = int(epoch.split('/')[0])
            reward = float(reward)
            epochs.append(epoch)
            rewards.append(reward)
   
    rewards = np.array(rewards)
    '''
    su = 0
    for i in range(len(rewards)):
        su += rewards[i]
        mean = su / (i + 1)
    print(ci)
    '''
    if 'DDQN' in name:
        '''
        for i, re in enumerate(rewards):
            if i > 0:
                rewards[i] = np.clip(rewards[i], rewards[i - 1] - 8, rewards[i - 1] + 8)
        '''
        plt.plot(epochs[:95:2], rewards[:95:2], label='DDQN')
    else:
        plt.plot(epochs[:95:2], rewards[:95:2], label='DQN')
plt.legend()
plt.title('Boxing-v0')
plt.xlabel('Epochs')
plt.ylabel('Rewards')
plt.savefig('p3_dqn.jpg')
#plt.show()
