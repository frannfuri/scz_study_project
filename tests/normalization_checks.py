clip_th = 2.2e-5
dmin = -clip_th
dmax = clip_th
y0 = dmin
y1 = dmax
plt.figure(figsize=(24, 2))
plt.plot(t, data_plot[2,:], linewidth=0.5)
plt.title('Origianl')
plt.ylim(y0, y1)
from extras import robust_z_score
plt.figure(figsize=(24, 2))
plt.plot(t, robust_z_score(data_plot[2,:]), linewidth=0.5)
plt.title('Normalized')
plt.ylim(-5, 5)
plt.show()
a=0