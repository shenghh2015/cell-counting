import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

from utils import file_load_save

model_root_folder = os.path.expanduser('~/dl-cells/dlct-framework/models/paper_models_w_cv')
# train_file1 = os.path.join(model_root_folder, 'date-6.21-buildMultiModel_U_net-mse-v-26-cross-5-batch-100-drop-None-lr-0.0001-nb_epochs-1-norm-True-t-0', 'training.pkl')     # the training file of the proposed method
# train_file2 = os.path.join(model_root_folder, 'date-6.25-buildModel_U_net-mse-v-26-cross-5-batch-100-drop-None-lr-0.0001-nb_epochs-1-norm-True-t-0', 'training.pkl')     # the training file of U-Net
# train_file3 = os.path.join(model_root_folder, 'date-6.23-buildModel_FCRN_A-mse-v-26-cross-5-batch-100-drop-None-lr-0.0001-nb_epochs-1-norm-True-t-0', 'training.pkl')     # the training file of FCRN-A
# data_v = 26

# train_file1 = os.path.join(model_root_folder, 'date-6.6-buildMultiModel_U_net-mse-v-27-cross-4-batch-100-drop-None-lr-0.0005-nb_epochs-1-norm-True-t-0', 'training.pkl')     # the training file of the proposed method
# train_file2 = os.path.join(model_root_folder, 'date-6.7-buildModel_U_net-mse-v-27-cross-4-batch-100-drop-None-lr-0.0005-nb_epochs-1-norm-True-t-0', 'training.pkl')     # the training file of U-Net
# train_file3 = os.path.join(model_root_folder, 'date-6.7-buildModel_FCRN_A-mse-v-27-cross-4-batch-100-drop-None-lr-0.0005-nb_epochs-1-norm-True-t-0', 'training.pkl')     # the training file of FCRN-A
# data_v = 27

# train_file1 = os.path.join(model_root_folder, 'date-7.2-buildMultiModel_U_net-mse-v-24-cross-5-batch-180-drop-None-lr-0.0001-nb_epochs-1-norm-True-t-0', 'training.pkl')     # the training file of the proposed method
# train_file2 = os.path.join(model_root_folder, 'date-6.10-buildModel_U_net-mse-v-24-cross-5-batch-100-drop-None-lr-0.005-nb_epochs-1-norm-True-t-0', 'training.pkl')     # the training file of U-Net
# train_file3 = os.path.join(model_root_folder, 'date-6.9-buildModel_FCRN_A-mse-v-24-cross-5-batch-100-drop-None-lr-0.005-nb_epochs-1-norm-True-t-0', 'training.pkl')     # the training file of FCRN-A
# data_v = 24

train_file1 = os.path.join(model_root_folder, 'date-7.2-buildMultiModel_U_net-mse-v-24-cross-3-batch-200-drop-None-lr-0.0001-nb_epochs-1-norm-True-t-0', 'training.pkl')     # the training file of the proposed method
train_file2 = os.path.join(model_root_folder, 'date-6.10-buildModel_U_net-mse-v-24-cross-3-batch-100-drop-None-lr-0.005-nb_epochs-1-norm-True-t-0', 'training.pkl')     # the training file of U-Net
train_file3 = os.path.join(model_root_folder, 'date-6.9-buildModel_FCRN_A-mse-v-24-cross-3-batch-100-drop-None-lr-0.005-nb_epochs-1-norm-True-t-0', 'training.pkl')     # the training file of FCRN-A
data_v = 24

# train_file1 = os.path.join(model_root_folder, 'date-6.15-buildMultiModel_U_net-mse-v-25-cross-3-batch-60-drop-None-lr-0.0001-nb_epochs-1-norm-True-t-0', 'training.pkl')     # the training file of the proposed method
# train_file2 = os.path.join(model_root_folder, 'date-6.18-buildModel_U_net-mse-v-25-cross-3-batch-60-drop-None-lr-0.0001-nb_epochs-1-norm-True-t-0', 'training.pkl')     # the training file of U-Net
# train_file3 = os.path.join(model_root_folder, 'date-6.16-buildModel_FCRN_A-mse-v-25-cross-3-batch-60-drop-None-lr-0.0001-nb_epochs-1-norm-True-t-0', 'training.pkl')     # the training file of FCRN-A
# data_v = 25

tr_loss1 = pickle.load(open(train_file1, 'rb'))['tr_loss']['ori']
te_loss1 = pickle.load(open(train_file1, 'rb'))['te_loss']['ori']

tr_loss2 = pickle.load(open(train_file2, 'rb'))['tr_loss']
te_loss2 = pickle.load(open(train_file2, 'rb'))['te_loss']

tr_loss3 = pickle.load(open(train_file3, 'rb'))['tr_loss']
te_loss3 = pickle.load(open(train_file3, 'rb'))['te_loss']

## save to matlab data
import scipy.io as sio
loss_dic = {}
loss_dic['our_tr_loss'] = np.array(tr_loss1)
loss_dic['our_te_loss'] = np.array(te_loss1)
loss_dic['unet_tr_loss'] = np.array(tr_loss2)
loss_dic['unet_te_loss'] = np.array(te_loss2)
loss_dic['fcrn_tr_loss'] = np.array(tr_loss3)
loss_dic['fcrn_te_loss'] = np.array(te_loss3)
sio.savemat('result_matlab/dataset_v{}.mat'.format(data_v), loss_dic)

## plot with python
color_dic = {'blue':'#0343df','red':'#e50000', 'green':'#15b01a', 
			 'violet':'#9a0eea','pink':'#fe01b1', 'orange':'#f97306', 'yellow':'#ffff14',
			 'maroon':'maroon'}
## 
plt.ion()
fig = plt.figure()
max_ = 500
min_ = 1
font_size = 14

plt.clf()
plt.plot(tr_loss1[min_:max_], '-', color=color_dic['blue'])
plt.plot(te_loss1[min_:max_], '-', color=color_dic['red'])
plt.plot(tr_loss2[min_:max_], '-', color=color_dic['green'])
plt.plot(te_loss2[min_:max_], '-', color=color_dic['pink'])
plt.plot(tr_loss3[min_:max_], '-', color=color_dic['violet'])
plt.plot(te_loss3[min_:max_], '-', color=color_dic['maroon'])
plt.xlim([0, max_])
plt.legend(['PriCNN+AuxCNNs:train','PriCNN+AuxCNNs:test','PriCNN-Only:train','PriCNN-Only:test','FCRN:train','FCRN:test'])


