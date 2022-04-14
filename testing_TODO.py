### CAMBIAR ENTEROOOO"""###
import torch
import pandas as pd
from collections import OrderedDict
from datasets import standardDataset, charge_all_data

from architectures import LinearHeadBENDR, BENDRClassification
# Since we are doing a lot of loading, this is nice to suppress some tedious information
import mne
mne.set_log_level(False)
### CAMBIAR TEST SPLIT !!!! y "how_much_healthy_controls" y dataset_name y subj_names y directory y th_clip y control_inits and chansconsider###
if __name__ == '__main__':
    #Data_samples, samples_filenames = load_eeg_data('./datasets/mmidb')
    #denied_subjs = [88, 90, 92, 100]
    
    # Hardcoded for 5 (4?) folds
    
    #test_splits = [ [[1,2,3],[12,13,14]], [[4,5,6],[1,2,3]],
    #                    [[7,8,9],[4,5,6]], [[10,11,12],[7,8,9]] ]
    test_splits = [ [list(range(21,25)),list(range(1,3))+list(range(28,31))],
                    [list(range(25,29)),list(range(3,8))], [list(range(1,5)), list(range(8,13))],
                    [list(range(5,9)),list(range(13,18))] ]

    samples_tlen = 20 #15 #20
    samples_overlap = 6 #5 #6
    arch  = 'bendr' #'linear' #'bendr'
    
    conc_test_splits = []
    for subset in test_splits:
        for f_id in range(len(subset[1])):
            subset[1][f_id] += 28
        conc_test_splits.append(subset[0]+subset[1])
    
    if arch == 'bendr':
        model_arch = BENDRClassification(targets=2, samples_len=samples_tlen*256, n_chn=20, encoder_h=512, contextualizer_hidden=3076, projection_head=False,
                                         new_projection_layers=0, dropout=0., trial_embeddings=None, layer_drop=0, keep_layers=None,
                                         mask_p_t=0.01, mask_p_c=0.005, mask_t_span=0.1, mask_c_span=0.1, multi_gpu=False, return_features=True)
        file_name = 'bendr_custom'
        
    elif arch == 'linear':
        model_arch = LinearHeadBENDR(n_targets=2, samples_len=samples_tlen*256, n_chn=20, encoder_h=512, projection_head=False,
                    enc_do=0.1, feat_do=0.1, pool_length=4, mask_p_t=0.01, mask_p_c=0.005, mask_t_span=0.05,
                    mask_c_span=0.1, classifier_layers=1, return_features=True)
        file_name = 'linear_custom'
        pass
    
    
    dataset_name = 'mdd_h_clean'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss().to(device)
    
    #Data_samples_bySubj = subdivide_according_to_subj(Data_samples, denied_subjs)
    #subj_names = list(range(1,29))
    subj_names = list(range(1,59))

    all_metrics = list()
    # Load dataset
    array_epochs_all_subjects = charge_all_data(directory='./datasets/mdd_h_clean', format_type='set', tlen=samples_tlen,
                                                overlap=samples_overlap, event_ids=['T1', 'T2'], th_clipping=2.4e-5,
                                                h_control_initials='H', chns_consider=['Fz', 'Cz', 'Pz'])
    
    print('Fold testing number 1...')
    model_arch.load_state_dict(torch.load('./results_new/best_model_f0.pt', map_location=device))
    model_arch.eval()
    model_arch = model_arch.to(device)
    with torch.no_grad():
        for i in conc_test_splits[0]:
            x_s = array_epochs_all_subjects[i-1][0]
            y_s = array_epochs_all_subjects[i-1][1]
            one_subj_dataset = standardDataset(x_s, y_s)
            one_subj_loader = torch.utils.data.DataLoader(one_subj_dataset, batch_size=1,
                                                shuffle=True, num_workers=0)
            metrics = OrderedDict()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in one_subj_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model_arch(inputs)
                _, preds = torch.max(outputs[0], 1)
                loss = criterion(outputs[0], labels)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(one_subj_loader)
            epoch_acc = running_corrects.double() / len(one_subj_loader)
            metrics['Person'] = subj_names[i-1]
            metrics['Dataset'] = dataset_name
            metrics['Accuracy'] = epoch_acc.item()
            metrics['loss'] = epoch_loss
            metrics['Fold'] = 1
            all_metrics.append(metrics)
        
    print('Fold testing number 2...')
    model_arch.load_state_dict(torch.load('./results_new/best_model_f1.pt', map_location=device))
    model_arch.eval()
    model_arch = model_arch.to(device)
    with torch.no_grad():
        for i in conc_test_splits[1]:
            x_s = array_epochs_all_subjects[i-1][0]
            y_s = array_epochs_all_subjects[i-1][1]
            one_subj_dataset = standardDataset(x_s, y_s)
            one_subj_loader = torch.utils.data.DataLoader(one_subj_dataset, batch_size=1,
                                                shuffle=True, num_workers=0)
            metrics = OrderedDict()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in one_subj_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model_arch(inputs)
                _, preds = torch.max(outputs[0], 1)
                loss = criterion(outputs[0], labels)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(one_subj_loader)
            epoch_acc = running_corrects.double() / len(one_subj_loader)
            metrics['Person'] = subj_names[i-1]
            metrics['Dataset'] = dataset_name
            metrics['Accuracy'] = epoch_acc.item()
            metrics['loss'] = epoch_loss
            metrics['Fold'] = 2
            all_metrics.append(metrics)
    
    print('Fold testing number 3...')
    model_arch.load_state_dict(torch.load('./results_new/best_model_f2.pt', map_location=device))
    model_arch.eval()
    model_arch = model_arch.to(device)
    with torch.no_grad():
        for i in conc_test_splits[2]:
            x_s = array_epochs_all_subjects[i-1][0]
            y_s = array_epochs_all_subjects[i-1][1]
            one_subj_dataset = standardDataset(x_s, y_s)
            one_subj_loader = torch.utils.data.DataLoader(one_subj_dataset, batch_size=1,
                                                shuffle=True, num_workers=0)
            metrics = OrderedDict()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in one_subj_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model_arch(inputs)
                _, preds = torch.max(outputs[0], 1)
                loss = criterion(outputs[0], labels)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(one_subj_loader)
            epoch_acc = running_corrects.double() / len(one_subj_loader)
            metrics['Person'] = subj_names[i-1]
            metrics['Dataset'] = dataset_name
            metrics['Accuracy'] = epoch_acc.item()
            metrics['loss'] = epoch_loss
            metrics['Fold'] = 3
            all_metrics.append(metrics)
            
    print('Fold testing number 4...')
    model_arch.load_state_dict(torch.load('./results_new/best_model_f3.pt', map_location=device))
    model_arch.eval()
    model_arch = model_arch.to(device)
    with torch.no_grad():
        for i in conc_test_splits[3]:
            x_s = array_epochs_all_subjects[i-1][0]
            y_s = array_epochs_all_subjects[i-1][1]
            one_subj_dataset = standardDataset(x_s, y_s)
            one_subj_loader = torch.utils.data.DataLoader(one_subj_dataset, batch_size=1,
                                                shuffle=True, num_workers=0)
            metrics = OrderedDict()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in one_subj_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model_arch(inputs)
                _, preds = torch.max(outputs[0], 1)
                loss = criterion(outputs[0], labels)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(one_subj_loader)
            epoch_acc = running_corrects.double() / len(one_subj_loader)
            metrics['Person'] = subj_names[i-1]
            metrics['Dataset'] = dataset_name
            metrics['Accuracy'] = epoch_acc.item()
            metrics['loss'] = epoch_loss
            metrics['Fold'] = 4
            all_metrics.append(metrics)
        
        
    
    
    Metrics_DataFrame = pd.DataFrame(all_metrics)#, index=False)
    Metrics_DataFrame.to_pickle("./" + file_name + ".pkl", protocol=4)
    