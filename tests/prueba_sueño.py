import mne
raw = mne.io.read_raw_edf('/home/jack/Descargas/SC4051E0-PSG.edf')
ans = mne.read_annotations('/home/jack/Descargas/SC4051EC-Hypnogram.edf')
ans.delete(-1)
raw.set_annotations(ans)
events = mne.events_from_annotations(raw)
epochs = mne.Epochs(raw, events[0], tmin=0, tmax=29/ raw.info['sfreq'], preload=True, decim=1,
                          baseline=None, reject_by_annotation=False)
a = 0
