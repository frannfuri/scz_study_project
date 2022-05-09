import torch
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew

outputs = torch.load('./results_new/scz_outputs_with_ssc_model_cpu_.pt')
score_W = []
for i in outputs:
    score_W.append(i[0][0].item())
plt.hist(score_W, bins=20)
plt.xlim((-1,9))
plt.ylim((0,5))
plt.title('Day11')
print(kurtosis(score_W, bias=False))
print(skew(score_W, bias=False))
plt.show()
a=0
