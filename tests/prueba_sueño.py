import mne
raw = mne.io.read_raw_edf('/home/jack/Descargas/data/SC4051E0-PSG.edf')
ans = mne.read_annotations('/home/jack/Descargas/data/SC4051EC-Hypnogram.edf')
ans.delete(-1)
raw.set_annotations(ans)
events = mne.events_from_annotations(raw)

raw2 = mne.io.read_raw_edf('/home/jack/Descargas/data/SC4051E0-PSG.edf')
ans2 = mne.read_annotations('/home/jack/Descargas/data/SC4051EC-Hypnogram.edf')
ans2.delete(-1)
raw2.set_annotations(ans2)
events2 = mne.events_from_annotations(raw2, {'Sleep stage 1': 1, 'Sleep stage 2': 2,
                                           'Sleep stage 3': 3, 'Sleep stage 4': 3,
                                           'Sleep stage R': 4, 'Sleep stage W': 0})
epochs2 = mne.Epochs(raw2, events2[0], tmin=0, tmax=29/ raw.info['sfreq'], preload=True, decim=1,
                          baseline=None, reject_by_annotation=False)
epochs = mne.Epochs(raw, events[0], tmin=0, tmax=29/ raw.info['sfreq'], preload=True, decim=1,
                          baseline=None, reject_by_annotation=False)
a = 0
