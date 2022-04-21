import numpy as np
import torch
import yaml
from datasets import charge_all_data, standardDataset
from sklearn.metrics import confusion_matrix
from architectures import BENDRClassification

#PARAMETERSSSS
model = 'bendr'
dataset = 'datasets/scz_decomp'

## HARDCODED !! ##
# Modify txt file first! Luego copiar y pegar
train0 = np.array([0, 1, 2, 4, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
                      , 21, 23, 24, 25, 26, 27, 29, 30, 31, 33, 35, 36, 38, 39, 40, 41, 42, 43
                      , 44, 45, 54, 55, 56, 57, 58, 59, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70
                      , 72, 73, 75, 76, 78, 79, 80, 81, 84, 86, 88, 89, 91, 92, 93, 94, 95, 96
                      , 97, 98, 99, 100, 104, 106, 108, 109, 110, 111, 112, 113, 115, 116, 117, 119, 120, 121
                      , 122, 123, 124, 126, 127, 128, 129, 131, 132, 133, 134, 135, 136, 137, 138, 139, 143, 145
                      , 146, 147, 149, 151, 153, 154, 156, 158, 159, 162, 165, 166, 167, 171, 173, 175, 176, 177
                      , 178, 179, 181, 182, 183, 184, 185, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 199
                      , 200, 201, 202, 204, 205, 206, 207, 208, 209, 210, 211, 214, 215, 216, 220, 221, 222, 223
                      , 224, 225, 226, 227, 229, 231, 232, 233, 234, 235, 236, 237, 238, 241, 242, 243, 244, 245
                      , 246, 248, 249, 250, 251, 254, 255, 256, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267
                      , 269, 270, 271, 272, 274, 275, 276, 277, 278, 280, 281, 282, 283, 284, 285, 286, 287, 288
                      , 289, 290, 291, 292, 293, 296, 297, 298, 299, 301, 302])
test0 = np.array([3, 6, 8, 22, 28, 32, 34, 37, 46, 47, 48, 49, 50, 51, 52, 53, 60, 71
                     , 74, 77, 82, 83, 85, 87, 90, 101, 102, 103, 105, 107, 114, 118, 125, 130, 140, 141
                     , 142, 144, 148, 150, 152, 155, 157, 160, 161, 163, 164, 168, 169, 170, 172, 174, 180, 186
                     , 187, 198, 203, 212, 213, 217, 218, 219, 228, 230, 239, 240, 247, 252, 253, 257, 268, 273
                     , 279, 294, 295, 300])
train1 = np.array([0, 2, 3, 4, 5, 6, 8, 11, 13, 14, 15, 16, 17, 19, 20, 22, 23, 24
                      , 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 43, 44, 46, 47, 48
                      , 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 62, 64, 65, 67, 68, 69
                      , 70, 71, 72, 74, 76, 77, 78, 79, 80, 81, 82, 83, 85, 87, 88, 90, 91, 95
                      , 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 113, 114, 115, 116, 117
                      , 118, 119, 122, 124, 125, 126, 128, 129, 130, 131, 132, 134, 137, 139, 140, 141, 142, 143
                      , 144, 145, 146, 147, 148, 149, 150, 151, 152, 155, 157, 160, 161, 162, 163, 164, 168, 169
                      , 170, 172, 174, 176, 177, 179, 180, 182, 184, 186, 187, 189, 190, 191, 192, 193, 194, 196
                      , 197, 198, 201, 202, 203, 204, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217
                      , 218, 219, 220, 222, 223, 227, 228, 229, 230, 232, 233, 234, 235, 236, 237, 238, 239, 240
                      , 242, 243, 244, 245, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260
                      , 261, 262, 263, 264, 266, 267, 268, 269, 270, 272, 273, 274, 276, 278, 279, 281, 283, 285
                      , 286, 289, 291, 292, 294, 295, 296, 297, 298, 300, 302])
test1 = np.array([1, 7, 9, 10, 12, 18, 21, 25, 26, 29, 41, 42, 45, 61, 63, 66, 73, 75
                     , 84, 86, 89, 92, 93, 94, 96, 110, 111, 112, 120, 121, 123, 127, 133, 135, 136, 138
                     , 153, 154, 156, 158, 159, 165, 166, 167, 171, 173, 175, 178, 181, 183, 185, 188, 195, 199
                     , 200, 205, 221, 224, 225, 226, 231, 241, 246, 265, 271, 275, 277, 280, 282, 284, 287, 288
                     , 290, 293, 299, 301])
train2 = np.array([1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 16, 17, 18, 21, 22, 25, 26
                      , 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 39, 41, 42, 43, 44, 45, 46, 47
                      , 48, 49, 50, 51, 52, 53, 54, 55, 57, 60, 61, 62, 63, 64, 65, 66, 67, 68
                      , 71, 72, 73, 74, 75, 76, 77, 79, 82, 83, 84, 85, 86, 87, 89, 90, 92, 93
                      , 94, 95, 96, 99, 101, 102, 103, 105, 107, 108, 109, 110, 111, 112, 113, 114, 118, 120
                      , 121, 123, 124, 125, 127, 128, 130, 131, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142
                      , 143, 144, 146, 148, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 163, 164
                      , 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 178, 180, 181, 182, 183, 185, 186
                      , 187, 188, 189, 190, 195, 196, 198, 199, 200, 201, 202, 203, 204, 205, 206, 212, 213, 215
                      , 216, 217, 218, 219, 220, 221, 222, 224, 225, 226, 228, 229, 230, 231, 236, 237, 238, 239
                      , 240, 241, 242, 243, 245, 246, 247, 248, 252, 253, 257, 258, 260, 262, 264, 265, 266, 267
                      , 268, 269, 271, 272, 273, 274, 275, 276, 277, 279, 280, 282, 283, 284, 286, 287, 288, 289
                      , 290, 291, 292, 293, 294, 295, 297, 299, 300, 301, 302])
test2 = np.array([0, 5, 13, 14, 15, 19, 20, 23, 24, 36, 38, 40, 56, 58, 59, 69, 70, 78
                     , 80, 81, 88, 91, 97, 98, 100, 104, 106, 115, 116, 117, 119, 122, 126, 129, 137, 145
                     , 147, 149, 162, 176, 177, 179, 184, 191, 192, 193, 194, 197, 207, 208, 209, 210, 211, 214
                     , 223, 227, 232, 233, 234, 235, 244, 249, 250, 251, 254, 255, 256, 259, 261, 263, 270, 278
                     , 281, 285, 296, 298])
train3 = np.array([0, 1, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 18, 19, 20, 21, 22
                      , 23, 24, 25, 26, 28, 29, 32, 34, 36, 37, 38, 40, 41, 42, 45, 46, 47, 48
                      , 49, 50, 51, 52, 53, 56, 58, 59, 60, 61, 63, 66, 69, 70, 71, 73, 74, 75
                      , 77, 78, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 96
                      , 97, 98, 100, 101, 102, 103, 104, 105, 106, 107, 110, 111, 112, 114, 115, 116, 117, 118
                      , 119, 120, 121, 122, 123, 125, 126, 127, 129, 130, 133, 135, 136, 137, 138, 140, 141, 142
                      , 144, 145, 147, 148, 149, 150, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163
                      , 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181
                      , 183, 184, 185, 186, 187, 188, 191, 192, 193, 194, 195, 197, 198, 199, 200, 203, 205, 207
                      , 208, 209, 210, 211, 212, 213, 214, 217, 218, 219, 221, 223, 224, 225, 226, 227, 228, 230
                      , 231, 232, 233, 234, 235, 239, 240, 241, 244, 246, 247, 249, 250, 251, 252, 253, 254, 255
                      , 256, 257, 259, 261, 263, 265, 268, 270, 271, 273, 275, 277, 278, 279, 280, 281, 282, 284
                      , 285, 287, 288, 290, 293, 294, 295, 296, 298, 299, 300, 301])
test3 = np.array([2, 4, 11, 16, 17, 27, 30, 31, 33, 35, 39, 43, 44, 54, 55, 57, 62, 64
                     , 65, 67, 68, 72, 76, 79, 95, 99, 108, 109, 113, 124, 128, 131, 132, 134, 139, 143
                     , 146, 151, 182, 189, 190, 196, 201, 202, 204, 206, 215, 216, 220, 222, 229, 236, 237, 238
                     , 242, 243, 245, 248, 258, 260, 262, 264, 266, 267, 269, 272, 274, 276, 283, 286, 289, 291
                     , 292, 297, 302])

def make_iterable_cv(tuples_array):
    for i in range(4):
        yield tuples_array[i]

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

is_first_rec = True
for rec in array_epochs_all_subjects:
    if is_first_rec:
        all_X = rec[0]
        all_y = rec[1]
        is_first_rec = False
    else:
        all_X = torch.cat((all_X, rec[0]), dim=0)
        all_y = torch.cat((all_y, rec[1]), dim=0)
all_dataset = standardDataset(all_X, all_y)

cv = make_iterable_cv([(train0, test0), (train1, test1), (train2, test2), (train3, test3)])
cfm = np.zeros((2,2))
f = 0
for _, test_data in cv:
    if model == 'bendr':
        model = BENDRClassification(targets=2, samples_len=samples_tlen * 256, n_chn=20, encoder_h=512,
                                        contextualizer_hidden=3076, projection_head=False,
                                        new_projection_layers=0, dropout=0., trial_embeddings=None, layer_drop=0,
                                        keep_layers=None,
                                        mask_p_t=0.01, mask_p_c=0.005, mask_t_span=0.1, mask_c_span=0.1,
                                        multi_gpu=False, return_features=True)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load('./results_new/best_model_f{}.pt'.format(f), map_location=device))
    model.eval()

    model = model.to(device)
    fold_X,  fold_y = all_dataset[test_data]
    outputs = model(fold_X)
    _, preds = torch.max(outputs[0], 1)
    cfm_f = confusion_matrix(fold_y, preds, normalize=True)
    print(cfm_f)
    cfm = cfm + cfm_f
    f += 1

cfm = cfm/4

a = 0