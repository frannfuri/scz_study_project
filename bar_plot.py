import numpy as np
import matplotlib.pyplot as plt
import torch


outputs = torch.load('./results_new/scz_outputs_with_ssc_model_cpu_.pt')
N = len(outputs)
p0 = [day[0][0].item() for day in outputs]
p1 = [day[0][1].item() for day in outputs]
p2 = [day[0][2].item() for day in outputs]
p3 = [day[0][3].item() for day in outputs]
p4 = [day[0][4].item() for day in outputs]


ind = np.arange(N*2, step=2)  # the x locations for the groups
width = 0.3       # the width of the bars

fig = plt.figure()
ax = fig.add_subplot(111)
rects0 = ax.bar(ind, p0, width, color='royalblue')
rects1 = ax.bar(ind+width, p1, width, color='seagreen')
rects2 = ax.bar(ind+width*2, p2, width, color='red')
rects3 = ax.bar(ind+width*3, p3, width, color='orange')
rects4 = ax.bar(ind+width*4, p4, width, color='gray')

# add some
ax.set_ylabel('Scores')
ax.set_ylim((-9,7))
ax.set_title('Scores of each window')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels([str(i) for i in list(range(N))])
ax.legend((rects0[0], rects1[0], rects2[0], rects3[0], rects4[0]), ('W', 'N1', 'N2',
                                                                     'N3', 'REM'))
ax.set_xlabel('Windows in time')

plt.show()
a = 0

