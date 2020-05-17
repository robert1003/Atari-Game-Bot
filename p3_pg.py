import matplotlib.pyplot as plt

for name in ['pg.log', 'pg_vr.log']:
    epochs = []
    rewards = []
    with open(name, 'r') as f:
        for line in f:
            _, epoch, _, _, _, reward, _, _, _, best_reward = line.split()
            epoch = int(epoch.split('/')[0])
            reward = float(reward)
            epochs.append(epoch)
            rewards.append(reward)
   
    if name == 'pg.log':
        plt.plot(epochs[:4000:10], rewards[:4000:10], label='vanilla')
    else:
        plt.plot(epochs[:4000:10], rewards[:4000:10], label='variance reduction')
plt.title('LunarLander-v2')
plt.xlabel('Epochs')
plt.ylabel('Rewards')
plt.legend()
plt.savefig('p3_pg.jpg')
#plt.show()
