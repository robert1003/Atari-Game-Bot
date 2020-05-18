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
   
a1, a2 = [], []
for _ in range(len(epochs) - 100):
    a1.append((_ + 1) * 10)
    a2.append(sum(rewards[_:_+100]) / 100)
plt.title('Deep Q-Learning')
plt.xlabel('Episode')
plt.ylabel('Rewards')
plt.plot(a1, a2)
plt.savefig('p1_dqn.jpg')
#plt.show()
