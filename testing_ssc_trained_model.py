import torch
import yaml
from datasets import charge_all_data, standardDataset
from architectures import BENDRClassification

#PARAMETERSSSS
model_type = 'bendr'
dataset = 'datasets/scz_decomp_single'
results_filename = 'new'

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
                                                chns_consider=data_settings['chns_to_consider'],
                                                had_annotations=data_settings['had_annotations'])
Xs = array_epochs_all_subjects[0][0]
outputs = []
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

for i in range(Xs.shape[0]):
    X = torch.unsqueeze(Xs[i,:,:], dim=0)
    X.to(device)
    outputs.append(model(X))

torch.save(outputs, './results_' + results_filename + '/scz_outputs_with_ssc_model_.pt')
torch.save(outputs.detach().cpu(), './results_' + results_filename + '/scz_outputs_with_ssc_model_cpu_.pt')