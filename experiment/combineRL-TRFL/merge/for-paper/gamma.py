import matplotlib.pyplot as plt
import numpy as np
 
gamma = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
 
hit_RLE = [0.4258, 0.4239, 0.425, 0.4268, 0.4273, 0.4242]
ndcg_RLE = [0.2618, 0.2607, 0.2615, 0.2628, 0.2618, 0.2607]

# plt.figure(figsize=(8, 8))
 
fig1, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(gamma, hit_RLE, 'b-')
ax2.plot(gamma, ndcg_RLE, 'r-')
 
ax1.set_xlabel(r'$\gamma$')

ax1.set_ylabel("Hit@10", color='b')
ax1.yaxis.label.set_color('b')
ax2.set_ylabel("NDCG@10", color='r')
ax2.yaxis.label.set_color('r')
 
plt.show()