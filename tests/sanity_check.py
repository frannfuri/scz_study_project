import torch
import os
import mne
import numpy as np
import matplotlib.pyplot as plt

from collections import OrderedDict

from matplotlib.collections import LineCollection

dataset_directory = "datasets/scz_decomp"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('---Using ' + str(device) + 'device---')

samples_tlen = 15
samples_overlap = 10
chns_to_consider = ['Fz', 'Cz', 'Pz']
clip_th = 2.4e-5 # 22uV heuristic
subj_to_check = 2
# TODO: APPLY CLIP TO RawEEG OBJECT

for root, _, files in os.walk(dataset_directory):
    j = 0
    for file in sorted(files):
        if file.endswith('set'):
            if j == subj_to_check:
                print('====================Processing subject number ' + str(j) + ' ('+file +')====================')
                if file.startswith('H'):
                    label = 0
                else:
                    label = 1
                new_raw2 = mne.io.read_raw_eeglab(os.path.join(root, file), preload=True)
                new_raw = mne.io.read_raw_eeglab(os.path.join(root, file))
                
                new_raw2.drop_channels(list(set(new_raw2.ch_names) - set(chns_to_consider)))
                new_raw.drop_channels(list(set(new_raw.ch_names) - set(chns_to_consider)))
                
                ##
                filename = new_raw.filenames[0].split('/')[-1]
                #event_ids = OrderedDict(['T1', 'T2'])
                #orig_sfreq = new_raw.info['sfreq']
                #new_sfreq = 256
                #ch_names = new_raw.ch_names
                #ch_list = []
                #for i in new_raw.ch_names:
                #    ch_list.append([i, '2'])
                #çh_list = np.array(ch_list, dtype='<U21')
                epochs =  mne.make_fixed_length_epochs(new_raw, id=label, duration=samples_tlen, overlap=samples_overlap)
                epochs.drop_bad()
                ##
                
                break
            j += 1

print()
print('Complete record with a total of %i timepoints (original freq of %i Hz), equivalent to %.2f seconds.'%(
        new_raw2._data.shape[1], orig_sfreq, new_raw2._data.shape[1]/orig_sfreq))
print()
print('Each sample has a length of %i seconds with a overlap of %i seconds, so there are obtained %i samples.'%(
        samples_tlen, samples_overlap, np.floor((new_raw2._data.shape[1]/orig_sfreq-(samples_tlen))/(samples_tlen-samples_overlap))+1))
if np.floor((new_raw2._data.shape[1]/orig_sfreq-(samples_tlen))/(samples_tlen-samples_overlap))+1 == epochs.events.shape[0]:
    print('The dimensions are consistent! ... for the moment.')
else:
    raise ValueError()

fig = plt.figure(figsize=(24, 9))
ax1 = fig.add_subplot(2,3,1)
ax2 = fig.add_subplot(2,3,2)
ax3 = fig.add_subplot(2,3,3)
ax4 = fig.add_subplot(2,1,2)
axes = [ax1, ax2, ax3, ax4]

# PLOT COMPLETE EEG RECORD
n_rows = last_chns_to_consider
datas = new_raw2._data
t_plot = new_raw2._data.shape[1]/orig_sfreq

data_plot = datas[datas.shape[0]-n_rows:]
t = t_plot * np.arange(data_plot.shape[1]) / data_plot.shape[1]
ticklocs = []
ax4.set_xlim(0,t_plot)
ax4.set_xticks(np.arange(t_plot))
#dmin = np.transpose(data_plot).min()
#dmax = np.transpose(data_plot).max()
dmin = -clip_th
dmax = clip_th 
dr = (dmax - dmin)*1.2 #* 0.7  # Crowd them a bit.
y0 = dmin
y1 = (n_rows - 1) * dr + dmax
ax4.set_ylim(y0, y1)


segs = []
for i in range(n_rows):
    segs.append(np.column_stack((t, np.clip(np.transpose(data_plot)[:, i],-abs(clip_th),abs(clip_th)))))
    ticklocs.append(i * dr)

offsets = np.zeros((n_rows, 2), dtype=float)
offsets[:, 1] = ticklocs

lines = LineCollection(segs, offsets=offsets, transOffset=None, linewidths=0.5)
ax4.add_collection(lines)

# Set the yticks to use axes coordinates on the y axis
ax4.set_yticks(ticklocs)
ax4.set_yticklabels(['Fz', 'Cz', 'Pz'])

ax4.set_xlabel('Time (s)')
ax4.set_title('Complete EEG record, Subject number %i' %(subj_to_check+1))

# PLOT EPOCHS 
ep1 = np.squeeze(epochs[0].get_data(list(range(last_chns_to_consider))))
ep2 = np.squeeze(epochs[-2].get_data(list(range(last_chns_to_consider))))
ep3 = np.squeeze(epochs[-1].get_data(list(range(last_chns_to_consider))))
t_ep_plot = ep1.shape[1]/orig_sfreq
t = t_ep_plot * np.arange(ep1.shape[1]) / ep1.shape[1]
ax1.set_xlim(0,t_ep_plot)
ax2.set_xlim(0,t_ep_plot)
ax3.set_xlim(0,t_ep_plot)
ax1.set_xticks(np.arange(t_ep_plot))
ax2.set_xticks(np.arange(t_ep_plot))
ax3.set_xticks(np.arange(t_ep_plot))
ax1.set_ylim(y0, y1)
ax2.set_ylim(y0, y1)
ax3.set_ylim(y0, y1)

ticklocs = []
segs = []
for i in range(n_rows):
    segs.append(np.column_stack((t, np.clip(np.transpose(ep1)[:, i],-abs(clip_th),abs(clip_th)))))
    ticklocs.append(i * dr)

offsets = np.zeros((n_rows, 2), dtype=float)
offsets[:, 1] = ticklocs

lines = LineCollection(segs, offsets=offsets, transOffset=None, linewidths=0.5)
ax1.add_collection(lines)

# Set the yticks to use axes coordinates on the y axis
ax1.set_yticks(ticklocs)
ax1.set_yticklabels(['Fz', 'Cz', 'Pz'])

ticklocs = []
segs = []
for i in range(n_rows):
    segs.append(np.column_stack((t, np.clip(np.transpose(ep2)[:, i],-abs(clip_th),abs(clip_th)))))
    ticklocs.append(i * dr)

offsets = np.zeros((n_rows, 2), dtype=float)
offsets[:, 1] = ticklocs

lines = LineCollection(segs, offsets=offsets, transOffset=None, linewidths=0.5)
ax2.add_collection(lines)

# Set the yticks to use axes coordinates on the y axis
ax2.set_yticks(ticklocs)
ax2.set_yticklabels(['Fz', 'Cz', 'Pz'])

ticklocs = []
segs = []
for i in range(n_rows):
    segs.append(np.column_stack((t, np.clip(np.transpose(ep3)[:, i],-abs(clip_th),abs(clip_th)))))
    ticklocs.append(i * dr)

offsets = np.zeros((n_rows, 2), dtype=float)
offsets[:, 1] = ticklocs

lines = LineCollection(segs, offsets=offsets, transOffset=None, linewidths=0.5)
ax3.add_collection(lines)

# Set the yticks to use axes coordinates on the y axis
ax3.set_yticks(ticklocs)
ax3.set_yticklabels(['Fz', 'Cz', 'Pz'])


plt.tight_layout()
#plt.show()h
# PLOT LONG SEGMENT OF THE  EEG RECORD
t_plot = 34
clip_th = 2.2e-5 # 22uV

fig2 = plt.figure(figsize=(24, 4))
ax = fig2.add_subplot(1,1,1)
data_plot = datas[datas.shape[0]-n_rows:]
#data_plot = data_plot[:, :int(t_plot*orig_sfreq)]
data_plot = data_plot[:, data_plot.shape[1]-int(t_plot*orig_sfreq):]
t = t_plot * np.arange(data_plot.shape[1]) / data_plot.shape[1]
ticklocs = []
ax.set_xlim(0,t_plot)
ax.set_xticks(np.arange(t_plot))
ax.set_ylim(y0, y1)

segs = []
for i in range(n_rows):
    segs.append(np.column_stack((t, np.clip(np.transpose(data_plot)[:, i],-abs(clip_th),abs(clip_th)))))
    ticklocs.append(i * dr)

offsets = np.zeros((n_rows, 2), dtype=float)
offsets[:, 1] = ticklocs

lines = LineCollection(segs, offsets=offsets, transOffset=None, linewidths=0.5)
ax.add_collection(lines)

# Set the yticks to use axes coordinates on the y axis
ax.set_yticks(ticklocs)
ax.set_yticklabels(['Fz', 'Cz', 'Pz'])

ax.set_xlabel('Time (s)')
ax.set_title('Last %i seconds of the EEG record' %(t_plot))
''''
ax.axvline(x=0, linestyle='dashed', c='r')
ax.text(0.1,0,'Begin ep1',rotation=90, c='r')
ax.axvline(x=20, linestyle='dashed', c='r')
ax.text(20.1,0,'End ep1',rotation=90, c='r')
ax.axvline(x=14, linestyle='dashed', c='g')
ax.text(14.1,0,'Begin ep2',rotation=90, c='g')
ax.axvline(x=34, linestyle='dashed', c='g')
ax.text(34.1,0,'End ep2',rotation=90, c='g')
'''''

plt.tight_layout()
#plt.show()


new_raw.drop_channels(new_raw.ch_names[:-n_rows])

mne.viz.plot_raw(new_raw, duration=new_raw2._data.shape[1]/orig_sfreq, block=False)

a=0
b=0