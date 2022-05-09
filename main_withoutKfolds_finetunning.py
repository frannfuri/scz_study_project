import argparse
import torch
import yaml
import os
import numpy as np
from datasets import charge_all_data, standardDataset
from architectures import MODEL_CHOICES, LinearHeadBENDR, BENDRClassification
from trainables import train_model
from torch.optim import lr_scheduler
from torch import nn
from sklearn.model_selection import train_test_split

train_IDs = [0,1,3,4,5,6,8,9] #day11,day13,day2,day6,day7,day9,day4,day5
test_IDs = [2,7]  #day1,day3
num_classes_pretrain = 2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tunes BENDER models.")
    parser.add_argument('model', choices=MODEL_CHOICES)
    # TODO: parser.add_argument('--subject-specific', action='store_true', help="Fine-tune on target subject alone.")
    # TODO: parser.add_argument('--mdl', action='store_true', help="Fine-tune on target subject using all extra data.")
    parser.add_argument('--freeze-encoder', action='store_true', help="Whether to keep the encoder stage frozen. "
                                                                      "Will only be done if not randomly initialized.")
    parser.add_argument('--random-init', action='store_true', help='Randomly initialized BENDR for comparison.')
    # TODO:
    parser.add_argument('--multi-gpu', action='store_true', help='Distribute BENDR over multiple GPUs')
    parser.add_argument('--num-workers', default=4, type=int, help='Number of dataloader workers.')
    parser.add_argument('--results-filename', default=None, help='What to name the spreadsheet produced with all '
                                                                 'final results.')
    parser.add_argument('--dataset-directory', default=None,
                        help='Where is the ubication of the data samples and the information '
                             'associated to them.')

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('---Using ' + str(device) + 'device---')
    os.makedirs('./results_' + args.results_filename, exist_ok=True)
    os.makedirs('./logs_' + args.results_filename, exist_ok=True)

    with open(args.dataset_directory + '/info.yml') as infile:
        data_settings = yaml.load(infile, Loader=yaml.FullLoader)

    # Data sample params
    samples_tlen = data_settings['tlen']
    samples_overlap = data_settings['overlap_len']

    # Load dataset
    array_epochs_all_subjects = charge_all_data(directory=args.dataset_directory,
                                                format_type=data_settings['format_type'],
                                                tlen=samples_tlen, overlap=samples_overlap,
                                                event_ids=data_settings['event_ids'],
                                                data_max = data_settings['data_max'],
                                                data_min = data_settings['data_min'],
                                                h_control_initials=data_settings['h_control_initials'],
                                                chns_consider=data_settings['chns_to_consider'],
                                                had_annotations=data_settings['had_annotations'])
    array_epochs_all_subjects_train = []
    array_epochs_all_subjects_test = []
    for i_train in train_IDs:
        array_epochs_all_subjects_train.append(array_epochs_all_subjects[i_train])
    for i_test in test_IDs:
        array_epochs_all_subjects_test.append(array_epochs_all_subjects[i_test])

    is_first_rec = True
    for rec in array_epochs_all_subjects_train:
        if is_first_rec:
            all_X = rec[0]
            all_y = rec[1]
            is_first_rec = False
        else:
            all_X = torch.cat((all_X, rec[0]), dim=0)
            all_y = torch.cat((all_y, rec[1]), dim=0)
    all_X_train = all_X
    all_y_train = all_y
    is_first_rec = True
    for rec in array_epochs_all_subjects_test:
        if is_first_rec:
            all_X = rec[0]
            all_y = rec[1]
            is_first_rec = False
        else:
            all_X = torch.cat((all_X, rec[0]), dim=0)
            all_y = torch.cat((all_y, rec[1]), dim=0)
    all_X_test = all_X
    all_y_test = all_y

    # Set fixed random number seed
    torch.manual_seed(28)

    # Train params
    lr = float(data_settings['lr'])
    num_epochs = data_settings['epochs']
    bs = data_settings['batch_size']
    ######################
    # Start print
    print('--------------------------------')
    np.savetxt('./logs_' + args.results_filename + '/train_ids.csv', train_IDs, delimiter=',')
    np.savetxt('./logs_' + args.results_filename + '/test_ids.csv', test_IDs, delimiter=',')
    #np.savetxt('./logs_' + args.results_filename + '/train_ids_' + str(fold) + '.csv', train_ids, delimiter=',')
    #np.savetxt('./logs_' + args.results_filename + '/test_ids_' + str(fold) + '.csv', test_ids, delimiter=',')
    # Sample elements randomly from a given list of ids, no replacement.
    #train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    #test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    print('TOTAL DATA {}'.format(len(all_y_train)+len(all_y_test)))
    #X_train, X_test, y_train, y_test = train_test_split(all_X, all_y, test_size=0.2, random_state=1, shuffle=True)
    #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1, shuffle=True)  # 0.25 x 0.8 = 0.2
    print('TARGET TRAIN 0/1: {}/{},  total: {}'.format(
        (all_y_train == 0).sum(), (all_y_train == 1).sum(), len(all_y_train)))
    print('TARGET VALID 0/1: {}/{},  total: {}'.format(
        (all_y_test == 0).sum(), (all_y_test == 1).sum(), len(all_y_test)))

    class_sample_count = np.array(
        [len(np.where(all_y_train == t)[0]) for t in np.unique(all_y_train)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in all_y_train])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler_train = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))

    class_sample_count = np.array(
        [len(np.where(all_y_test == t)[0]) for t in np.unique(all_y_test)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in all_y_test])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler_val = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))

    # Define data loaders for training and testing data in this fold
    trainloader = torch.utils.data.DataLoader(
                standardDataset(all_X_train, all_y_train),
                batch_size=bs, sampler=sampler_train)
    # It is validloader actually, but for simplicity keeps its last variable name
    testloader = torch.utils.data.DataLoader(
                standardDataset(all_X_test, all_y_test),
                batch_size=bs, sampler=sampler_val)
    dataloaders = {'train': trainloader, 'val': testloader}

    dataset_sizes = {x: len(dataloaders[x])*bs for x in ['train', 'val']}

    # MODEL
    if args.model == MODEL_CHOICES[0]:
        model = BENDRClassification(targets=num_classes_pretrain, samples_len=samples_tlen * 256, n_chn=20, encoder_h=512,
                                    contextualizer_hidden=3076, projection_head=False,
                                    new_projection_layers=0, dropout=0., trial_embeddings=None, layer_drop=0,
                                    keep_layers=None,
                                    mask_p_t=0.01, mask_p_c=0.005, mask_t_span=0.1, mask_c_span=0.1,
                                    multi_gpu=False, return_features=True)
    else:
        model = LinearHeadBENDR(n_targets=num_classes_pretrain, samples_len=samples_tlen * 256, n_chn=20, encoder_h=512,
                                projection_head=False,
                                enc_do=0.1, feat_do=0.4, pool_length=4, mask_p_t=0.01, mask_p_c=0.005,
                                mask_t_span=0.05,
                                mask_c_span=0.1, classifier_layers=1, return_features=True)

        #model = model.to(device)
    if args.multi_gpu:
        model = nn.DataParallel(model)
    if not args.random_init:
        model.load_whole_pretrained_modules('./results_new/best_model_.pt', freeze_encoder=True,
                                            freeze_contextualizer=True, freeze_position_conv=True,
                                            freeze_mask_replacement=True, device=device)
    model.make_new_classification_layer(numb_of_targets=data_settings['num_cls'])

    model = model.to(device)

    #
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    sched = lr_scheduler.OneCycleLR(optimizer, lr, epochs=num_epochs, steps_per_epoch=len(dataloaders['train']),
                                    pct_start=0.3, last_epoch=-1)

    best_model, acc_curves, loss_curves, train_log, valid_log = train_model(model, criterion, optimizer, sched,
                                                                            dataloaders, dataset_sizes, device,
                                                                            num_epochs)
    train_log.to_pickle("./logs_{}/train_log_.pkl".format(args.results_filename), protocol=4)
    valid_log.to_pickle("./logs_{}/valid_log_.pkl".format(args.results_filename), protocol=4)
    torch.save(best_model.state_dict(), './results_{}/best_model_.pt'.format(args.results_filename))
    torch.save(loss_curves, './results_{}/loss_curves_.pt'.format(args.results_filename))
    torch.save(acc_curves, './results_{}/acc_curves_.pt'.format(args.results_filename))
    #torch.save(X_test, './results_{}/X_test_.pt'.format(args.results_filename))
    #torch.save(y_test, './results_{}/y_test_.pt'.format(args.results_filename))