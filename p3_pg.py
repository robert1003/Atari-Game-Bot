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
   
    a1, a2 = [], []
    for _ in range(len(epochs) - 200):
        a1.append((_ + 1) * 10)
        a2.append(sum(rewards[_:_+100]) / 100)

    if name == 'pg.log':
        #plt.plot(epochs[:4000:10], rewards[:4000:10], label='vanilla')
        plt.plot(a1[:4000], a2[:4000], label='vanilla')
    else:
        #plt.plot(epochs[:4000:10], rewards[:4000:10], label='variance reduction')
        plt.plot(a1[:4000], a2[:4000], label='variance reduction')
plt.title('LunarLander-v2')
plt.xlabel('Epochs')
plt.ylabel('Rewards')
plt.legend()
plt.savefig('p3_pg.jpg')
#plt.show()
