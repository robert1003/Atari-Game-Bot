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
   
plt.title('Policy Gradients')
plt.xlabel('Epochs')
plt.ylabel('Rewards')
plt.plot(epochs, rewards)
plt.savefig('p1_pg.jpg')
#plt.show()
