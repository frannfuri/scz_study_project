import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from datasets import charge_all_data, standardDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from architectures import BENDRClassification

#PARAMETERSSSS
model_type = 'bendr'
dataset = 'datasets/scz_decomp'
results_filename = 'new'

test0 = np.genfromtxt('./logs_' + results_filename + '/test_ids.csv', delimiter=',', dtype='int16')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
with open(dataset + '/info.yml') as infile:
    data_settings = yaml.load(infile, Loader=yaml.FullLoader)

# Data sample params
samples_tlen = data_settings['tlen']
samples_overlap = data_settings['overlap_len']

# Load dataset
array_epochs_all_subjects = charge_all_data(directory=dataset,
                                            format_type=data_settings['format_type'],
                                            tlen=samples_tlen, overlap=samples_overlap,
                                            event_ids=data_settings['event_ids'],
                                            data_max = data_settings['data_max'],
                                            data_min = data_settings['data_min'],
                                            h_control_initials=data_settings['h_control_initials'],
                                            chns_consider=data_settings['chns_to_consider'])
array_epochs_all_subjects_test = []
for test_id in test0:
    array_epochs_all_subjects_test.append(array_epochs_all_subjects[test_id])
is_first_rec = True
for rec in array_epochs_all_subjects_test:
    if is_first_rec:
        all_X = rec[0]
        all_y = rec[1]
        is_first_rec = False
    else:
        all_X = torch.cat((all_X, rec[0]), dim=0)
        all_y = torch.cat((all_y, rec[1]), dim=0)
test_dataset = standardDataset(all_X, all_y)

if model_type == 'bendr':
    model = BENDRClassification(targets=2, samples_len=samples_tlen * 256, n_chn=20, encoder_h=512,
                                    contextualizer_hidden=3076, projection_head=False,
                                    new_projection_layers=0, dropout=0., trial_embeddings=None, layer_drop=0,
                                    keep_layers=None,
                                    mask_p_t=0.01, mask_p_c=0.005, mask_t_span=0.1, mask_c_span=0.1,
                                    multi_gpu=False, return_features=True)
model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load('./results_new/best_model.pt'.format(f), map_location=device))
model.eval()

model = model.to(device)
testloader = torch.utils.data.DataLoader(test_dataset,
                                         batch_size=len(test_dataset),
                                         shuffle=True)
c = 0
for x,y in testloader:
    print('loop:')
    print(c)
    c += 1
    x = x.to(device)
    y = y.to(device)
    outputs = model(x)
    _, preds = torch.max(outputs[0], 1)
    cfm_f = confusion_matrix(y.detach().cpu(), preds.detach().cpu(), normalize='all')
    print(cfm_f)
disp = ConfusionMatrixDisplay(cfm_f)
disp.plot()
plt.show()

a = 0