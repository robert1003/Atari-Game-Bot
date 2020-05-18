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
    a1, a2 = [], []
    for _ in range(len(epochs) - 100):
        a1.append((_ + 1) * 10)
        a2.append(sum(rewards[_:_+100]) / 100)
    print(len(a1), len(a2))
    if 'DDQN' in name:
        '''
        for i, re in enumerate(rewards):
            if i > 0:
                rewards[i] = np.clip(rewards[i], rewards[i - 1] - 8, rewards[i - 1] + 8)
        '''
        #plt.plot(epochs[:566:10], rewards[:566:10], label='DDQN')
        plt.plot(a1[:460], a2[:460], label='DDQN')
    else:
        #plt.plot(epochs[:566:10], rewards[:566:10], label='DQN')
        plt.plot(a1[:460], a2[:460], label='DQN')
plt.legend()
plt.title('Boxing-v0')
plt.xlabel('Epochs')
plt.ylabel('Rewards')
plt.savefig('p3_dqn.jpg')
#plt.show()
