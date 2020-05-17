import matplotlib.pyplot as plt

epochs = []
rewards = []
with open('dqn_gamma_0.9.log', 'r') as f:
    for line in f:
        _, epoch, _, _, _, _, _, _, reward, _, _, _, _, _, _ = line.split()
        epoch = int(epoch)
        reward = float(reward)
        epochs.append(epoch)
        rewards.append(reward)
   
plt.title('Deep Q-Learning')
plt.xlabel('Episode')
plt.ylabel('Rewards')
plt.plot(epochs, rewards)
plt.savefig('p1_dqn.jpg')
#plt.show()
