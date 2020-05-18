import matplotlib.pyplot as plt

epochs = []
rewards = []
with open('pg.log', 'r') as f:
    for line in f:
        _, epoch, _, _, _, reward, _, _, _, best_reward = line.split()
        epoch = int(epoch.split('/')[0])
        reward = float(reward)
        epochs.append(epoch)
        rewards.append(reward)
   
a1, a2 = [], []
for _ in range(len(epochs) - 100):
    a1.append((_ + 1) * 10)
    a2.append(sum(rewards[_:_+100]) / 100)
plt.title('Policy Gradients')
plt.xlabel('Epochs')
plt.ylabel('Rewards')
plt.plot(a1, a2)
plt.savefig('p1_pg.jpg')
#plt.show()
