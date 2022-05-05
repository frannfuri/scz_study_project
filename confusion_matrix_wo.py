import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from datasets import charge_all_data, standardDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score, accuracy_score
from architectures import BENDRClassification


#PARAMETERSSSS
model_type = 'bendr'
dataset = 'datasets/sleep-cassette'
results_filename = 'new'
X_test = torch.load('./results_' + results_filename + '/X_test_.pt')
y_test = torch.load('./results_' + results_filename + '/y_test_.pt')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
with open(dataset + '/info.yml') as infile:
    data_settings = yaml.load(infile, Loader=yaml.FullLoader)

# Data sample params
samples_tlen = data_settings['tlen']
samples_overlap = data_settings['overlap_len']

# Load dataset
#array_epochs_all_subjects = charge_all_data(directory=args.dataset_directory,
#                                                format_type=data_settings['format_type'],
#                                                tlen=samples_tlen, overlap=samples_overlap,
#                                                event_ids=data_settings['event_ids'],
#                                                data_max = data_settings['data_max'],
#                                                data_min = data_settings['data_min'],
#                                                h_control_initials=data_settings['h_control_initials'],
#                                                chns_consider=data_settings['chns_to_consider'],
#                                                had_annotations=data_settings['had_annotations'])
test_dataset = standardDataset(X_test, y_test)

if model_type == 'bendr':
    model = BENDRClassification(targets=data_settings['num_cls'], samples_len=samples_tlen * 256, n_chn=20, encoder_h=512,
                                    contextualizer_hidden=3076, projection_head=False,
                                    new_projection_layers=0, dropout=0., trial_embeddings=None, layer_drop=0,
                                    keep_layers=None,
                                    mask_p_t=0.01, mask_p_c=0.005, mask_t_span=0.1, mask_c_span=0.1,
                                    multi_gpu=False, return_features=True)
model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load('./results_new/best_model_.pt', map_location=device))
model.eval()

model = model.to(device)
testloader = torch.utils.data.DataLoader(test_dataset,
                                         batch_size=64,
                                         shuffle=True)
c = 0
bac = 0
acc = 0
cfm_b = np.zeros((5,5))
for x,y in testloader:
    print('loop:')
    print(c)
    c += 1
    x = x.to(device)
    y = y.to(device)
    outputs = model(x)
    _, preds = torch.max(outputs[0], 1)
    cfm_b += confusion_matrix(y.detach().cpu(), preds.detach().cpu(), normalize='all')
    bac += balanced_accuracy_score(y.detach().cpu(), preds.detach().cpu())
    acc += accuracy_score(y.detach().cpu(), preds.detach().cpu())
disp = ConfusionMatrixDisplay(cfm_b/len(testloader))
disp.plot()
plt.show()
print('BAC: ' + str(bac/len(testloader)))
print('ACC: ' + str(acc/len(testloader)))

a = 0