import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 1, figsize=(10, 7))
for i, gamma in enumerate([0.1, 0.3, 0.5, 0.7, 0.9, 0.99]):
    epochs = []
    rewards = []
    with open('dqn_gamma_{}.log'.format(gamma), 'r') as f:
        for line in f:
            _, epoch, _, _, _, _, _, _, reward, _, _, _, _, _, _ = line.split()
            epoch = int(epoch)
            reward = float(reward)
            epochs.append(epoch)
            rewards.append(reward)
       

    print(len(epochs), len(rewards))
    a1, a2 = [], []
    for _ in range(len(epochs) - 100):
        a1.append((_ + 1) * 10)
        a2.append(sum(rewards[_:_+100]) / 100)

    #axs.plot(epochs[:3500:25], rewards[:3500:25], label='{}'.format(gamma))
    axs.plot(a1[:3500], a2[:3500], label='{}'.format(gamma))
    axs.set_xlabel('Episode')
    axs.set_ylabel('Rewards')
    axs.legend()
'''
for i, gamma in enumerate([0.9, 0.95, 0.96, 0.97, 0.98, 0.99]):
    epochs = []
    rewards = []
    with open('dqn_gamma_{}.log'.format(gamma), 'r') as f:
        for line in f:
            _, epoch, _, _, _, _, _, _, reward, _, _, _, _, _, _ = line.split()
            epoch = int(epoch)
            reward = float(reward)
            epochs.append(epoch)
            rewards.append(reward)
       
    print(len(epochs), len(rewards))
    axs[1].plot(epochs[:3500:50], rewards[:3500:50], label='{}'.format(gamma))
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Rewards')
    axs[1].legend()
'''
plt.savefig('p2.jpg', bbox_inches='tight')
#plt.show()
