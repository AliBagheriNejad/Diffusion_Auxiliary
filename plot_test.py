import json
import numpy as np
import matplotlib.pyplot as plt
with open('grad.json', 'r') as file:
    grad_dic = json.load(file)
with open('weight.json', 'r') as file:
    weight_dic = json.load(file)
p_keys = list(grad_dic['0'].keys())

get_vals = lambda x, k: [x[i][k] for i in x.keys()]
lim = p_keys[-10:]

for k in lim:
    # if ('bn' not in k) and ('bias' not in k) and ('u_1' in k) or ('u_2' in k):
        plt.plot(get_vals(weight_dic, k), label=k)
plt.legend()
plt.title('Weights')
plt.figure()
for k in lim:
    # if ('bn' not in k) and ('bias' not in k) and ('u_1' in k) or ('u_2' in k):
        plt.plot(get_vals(grad_dic, k), label=k)
plt.legend()
# plt.ylim([-25e-5, 25e-5])
plt.title('Gradients')
plt.show()
