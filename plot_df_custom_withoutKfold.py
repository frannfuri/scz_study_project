import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == '__main__':
    n_epochs = 15
    f1score_available = True
    # Curves my implementation #2 (Dataframe)
    path='./logs_new/'
    a0 = pd.read_pickle(path + 'train_log_.pkl')
    b0 = pd.read_pickle(path + 'valid_log_.pkl')
    tr_loss_f0_ = []
    val_loss_f0_ = []
    for i in range(n_epochs):
        tr_loss_f0_.append(a0[a0['epoch']==(i)].mean()['loss'])
        val_loss_f0_.append(b0[b0['epoch']==(i)].mean()['loss'])
        
    all_train_mean_loss_ = np.array(tr_loss_f0_)
    all_val_mean_loss_ = np.array(val_loss_f0_)

    plt.figure()
    plt.plot(all_train_mean_loss_, label='Mean train', lw=2)
    plt.plot(all_val_mean_loss_, label='Mean val', lw=2)
    plt.title('Training loss ')
    plt.grid()
    plt.legend(loc='best')
    
    
    
    tr_loss_f0_ = []
    val_loss_f0_ = []
    for i in range(n_epochs):
        tr_loss_f0_.append(a0[a0['epoch']==(i)].mean()['accuracy'])
        val_loss_f0_.append(b0[b0['epoch']==(i)].mean()['accuracy'])
    all_train_mean_loss_ = np.array(tr_loss_f0_)
    all_val_mean_loss_ = np.array(val_loss_f0_)

    plt.figure()
    plt.plot(all_train_mean_loss_, label='Mean train', lw=2)
    plt.plot(all_val_mean_loss_, label='Mean val', lw=2)
    plt.title('Training accuracy')
    plt.grid()
    plt.legend(loc='best')
    
    if f1score_available:
        tr_loss_f0_ = []
        val_loss_f0_ = []
        for i in range(n_epochs):
            tr_loss_f0_.append(a0[a0['epoch']==(i)].mean()['f1score'])
            val_loss_f0_.append(b0[b0['epoch']==(i)].mean()['f1score'])

        all_train_mean_loss_ = np.array(tr_loss_f0_)
        all_val_mean_loss_ = np.array(val_loss_f0_)
    
        plt.figure()
        plt.plot(all_train_mean_loss_, label='F1 score train', lw=2)
        plt.grid()
        plt.plot(all_val_mean_loss_, label='F1 score val', lw=2)
        plt.title('Training F1score (mean of the fold-iterations)')
        plt.legend(loc='best')
        
        #plt.ylim([0.0, 1.00])
        
        tr_loss_f0_ = []
        val_loss_f0_ = []
        for i in range(n_epochs):
            tr_loss_f0_.append(a0[a0['epoch']==(i)].mean()['preciss'])
            val_loss_f0_.append(b0[b0['epoch']==(i)].mean()['preciss'])

            
        all_train_mean_loss_ = np.array(tr_loss_f0_)
        all_val_mean_loss_ = np.array(val_loss_f0_)
    
        plt.plot(all_train_mean_loss_, label='Precission train', lw=1)
        plt.plot(all_val_mean_loss_, label='Precission val', lw=1)
        plt.grid()
        plt.legend(loc='best')

        tr_loss_f0_ = []
        val_loss_f0_ = []
        for i in range(n_epochs):
            tr_loss_f0_.append(a0[a0['epoch']==(i)].mean()['recall'])
            val_loss_f0_.append(b0[b0['epoch']==(i)].mean()['recall'])

        all_train_mean_loss_ = np.array(tr_loss_f0_)
        all_val_mean_loss_ = np.array(val_loss_f0_)
    
        plt.plot(all_train_mean_loss_, label='Recall train', lw=1)
        plt.plot(all_val_mean_loss_, label='Recall val', lw=1)
        plt.grid()
        plt.legend(loc='best')
        plt.show()
        a = 0
        