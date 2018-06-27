import numpy as np
import matplotlib.pyplot as plt

data = np.load('policy-gradient-dynamic-baseline_True-discount_0.99-epsilon_0.01-lr_critic_0.5-lr_intra_0.25-lr_term_100.0-nepisodes_1000-noptions_4-nruns_100-nsteps_1000-primitive_False-temperature_0.5.npy')
data = np.mean(data, axis=0)
step = data[:,0]
avgduration = data[:,1]

plt.subplot(2,1,1)
plt.plot(step)
plt.title('The average step length.')

plt.subplot(2,1,2)
plt.plot(avgduration)
plt.title('The average duration of each option.')
plt.show()