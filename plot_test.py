import json
import numpy as np
import matplotlib.pyplot as plt
with open('F:/thesis/Articles/2nd/diffusion/mlruns/772346583013907213/6d854691aec442cb897f929dcebad882/artifacts/artifacts/grad.json', 'r') as file:
    weight_dic = json.load(file)
with open('F:/thesis/Articles/2nd/diffusion/mlruns/772346583013907213/7b35b6a8e5d14290a2d50bfdfe74d7d8/artifacts/artifacts/grad.json', 'r') as file:
    grad_dic = json.load(file)
p_keys = list(grad_dic['0'].keys())

get_vals = lambda x, k: [x[i][k] for i in x.keys()]
lim = p_keys[-10:]
plt.subplot(2,1,1)
for k in lim:
    # if ('bn' not in k) and ('bias' not in k) and ('u_1' in k) or ('u_2' in k):
        plt.plot(get_vals(weight_dic, k), label=k)
plt.legend()
plt.title('different lr')
plt.subplot(2,1,2)
for k in lim:
    # if ('bn' not in k) and ('bias' not in k) and ('u_1' in k) or ('u_2' in k):
        plt.plot(get_vals(grad_dic, k), label=k)
plt.legend()
# plt.ylim([-25e-5, 25e-5])
plt.title('high lr')
plt.show()
