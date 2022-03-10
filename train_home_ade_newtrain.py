#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import torch
from torch import nn, optim
from resnet_model import Clf, ResNet, Decoder
import util
import numpy as np
import torch.nn.functional as F
import os
import shutil
from data_loader_officehome import OfficeHome
import time
np.random.seed(3)
torch.manual_seed(4)
#(0,1), (1,2), (2,3) (3,4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
parser = argparse.ArgumentParser(description='Hyperparameter alpha and beta')
parser.add_argument("--alpha", type=float,nargs='+', help="weight list for barycenter")
parser.add_argument("--beta", type=float, nargs='+', help="weight list for reconstruction")
parser.add_argument("--bs", type=int, default=32, help="batch size")
parser.add_argument("--itr", type=int, default=110, help="iteration based on batch size")
parser.add_argument("--epochs", type=int, default=40, help="training epochs")
parser.add_argument("--dropout", type=float, default=0, help="dropout ratio")
parser.add_argument("--blur", type=float, default=0.5, help="blur for WB")

opt = parser.parse_args()
print(opt)

# home files name is 'Art', 'Clipart', 'Product', 'Real'
test_file_list = ['Art', 'Clipart', 'Product', 'Real']

num_epochs = opt.epochs
batch_size = opt.bs
iteration = opt.itr
dropout = opt.dropout
blur= opt.blur
num_dirac = 100

learning_rate = 5e-5
alpha_list = opt.alpha
beta_list  = opt.beta # [0.001, 1] is not good


def save_checkpoint(state, is_best, filename,destination_name):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, destination_name)

'''
The loss of autoencoder should be consist of three parts: clf loss and wasserstein barycenter loss and
reconstruction loss
We will need an extra classifier for classification, the wasserstein barycenter can be used for the latent code directly.
'''
def train_model(ae_model,clf_model, d_model, generator,
                val_data, num_epochs, batch_size, alpha, beta, blur,num_dirac):
    history = {'train_losses':[],'train_clf_loss': [], 'train_bary_loss':[],'train_recon_loss':[],
               'val_acc':[], 'val_losses':[], 'test_acc':[], 'test_losses':[]}
    best_acc = 0
    for epoch in range(num_epochs):
        ae_model.train()
        clf_model.train()
        d_model.train()
        # for history
        total_loss = []
        train_clf_loss = []
        train_bary_loss = []
        train_recon_loss= []
        # criteria for classification loss
        clf_loss = nn.CrossEntropyLoss().to(device)
        recon_loss = nn.MSELoss().to(device)

        for batch_idx in range(iteration):
            batch_data, batch_label, batch_source = next(generator)

            latent_code = ae_model(batch_data)
            pred = clf_model(latent_code)

            loss_clf = clf_loss(pred, batch_label)
            loss_recon = recon_loss(batch_data,d_model(latent_code)) * beta
            loss_bary = util.wass_loss(latent_code, batch_source, blur=blur, num_dirac=num_dirac, device=device) * alpha
            # Total loss
            losses = loss_clf
            losses += loss_bary
            losses += loss_recon

            optimizer_ae.zero_grad()
            losses.backward()
            optimizer_ae.step()

            total_loss.append(losses.item())
            train_clf_loss.append(loss_clf.item())
            train_bary_loss.append(loss_bary.item())
            train_recon_loss.append(loss_recon.item())

        # for history
        total_loss = np.average(total_loss)
        train_clf_loss = np.average(train_clf_loss)
        train_bary_loss = np.average(train_bary_loss)
        train_recon_loss = np.average(train_recon_loss)


        acc1, one_val_loss = validation(ae_model, clf_model, val_data, batch_size)
        # save two checkpoint, one for lowest loss, one for max accuracy
        is_best_acc = acc1 > best_acc

        best_acc = max(acc1, best_acc)

        check_filename_loss = check_folder+'epoch' + str(num_epochs) + 'lr' + str(learning_rate) \
                + 'bs' + str(batch_size) +  'blur' + str(blur) + 'alpha' + str(alpha) +'beta'+ str(beta) +test_file + 'loss_checkpoint.pth.tar'
        dest_filename_loss = check_folder + 'epoch' + str(num_epochs) + 'lr' + str(learning_rate) \
                + 'bs' + str(batch_size)+ 'blur' + str(blur) + 'alpha' + str(alpha) +'beta'+ str(beta) +test_file + 'loss_model_best.pth.tar'

        save_checkpoint({
                'epoch': epoch + 1,
                'alpha': alpha,
                'beta': beta,
                'ae_state_dict': ae_model.state_dict(),
                'clf_state_dict': clf_model.state_dict(),
                'de_state_dict':d_model.state_dict(),
                'best_acc': best_acc
            }, is_best_acc, check_filename_loss, dest_filename_loss)
        if epoch % 1 == 0:
            test_acc, test_loss = validation(ae_model, clf_model, test_data, batch_size)
            history['test_losses'].append(test_loss)
            history['test_acc'].append(test_acc)
            print('Epoch ', epoch)
            # print('Training Loss: {:f}, Clf loss: {:f}, Wasserstein loss: {:f}'.format(total_loss, train_clf_loss, train_bary_loss))

            print('Training Loss: {:f}, Clf loss: {:f}, Wasserstein loss: {:f}, Recon loss: {:f},'.format(total_loss, train_clf_loss, train_bary_loss, train_recon_loss))
            print ('Val Loss: {:f}, Acc: {:f}'.format(one_val_loss,acc1))
            print ('Test Loss: {:f}, Test Acc: {:f}'.format(test_loss,test_acc))

        # save all information
        history['train_losses'].append(total_loss)
        history['train_clf_loss'].append(train_clf_loss)
        history['train_bary_loss'].append(train_bary_loss)
        history['train_recon_loss'].append(train_recon_loss)
        history['val_losses'].append(one_val_loss)
        history['val_acc'].append(acc1)

        # history['pred_res'].append(clf_model(ae_model.encode(X_test.detach()).data))
    # history['test_lb'].append(y_validation.data)
    return ae_model, clf_model, history



def validation(ae_model, clf_model,data_lb , batch_size):
    ae_model.eval()
    clf_model.eval()
    total_loss = []
    correct = []
    data, label = data_lb
    with torch.no_grad():
        for i in range(0, data.size()[0], batch_size):
            batch_data = data[i:i+batch_size]
            batch_label = label[i:i+batch_size]

            latent_code = ae_model(batch_data)
            y_pred = clf_model(latent_code)
            acc = util.accuracy(y_pred, batch_label)
            correct.append(acc)
            loss = F.cross_entropy(y_pred, batch_label)
            total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        correct = np.sum(correct)/label.size(0)
        return correct, total_loss

def get_result(check_path, test_data, batch_size, run_time):
    clf_model_test = Clf(out_features=65).to(device)
    ae_model_test = ResNet(dropout).to(device)
    d_model_test = Decoder().to(device)

    checkpoint_eval = torch.load(check_path)
    clf_model_test.load_state_dict(checkpoint_eval['clf_state_dict'])
    ae_model_test.load_state_dict(checkpoint_eval['ae_state_dict'])
    d_model_test.load_state_dict(checkpoint_eval['de_state_dict'])
    alpha  = checkpoint_eval['alpha']
    beta = checkpoint_eval['beta']
    best_criterion = checkpoint_eval['best_acc']

    clf_model_test.eval()
    ae_model_test.eval()
    test_acc, test_loss = validation(ae_model_test, clf_model_test,test_data, batch_size)
    with open(history_folder + 'results_' + 'lr' + str(learning_rate) + 'bs' + str(batch_size)+'epoch' + str(num_epochs) + 'alpha' + str(alpha) + 'beta' + str(beta)+'itr'+str(iteration)+'drop'+str(dropout)+'blur'+str(blur)+ '.txt', 'a+') as f:
        f.write('\n')
        f.write(test_file + ' lr' + str(learning_rate) + ' bs' + str(batch_size)+' epoch' + str(num_epochs))
        f.write(' latent_dim_blur ' + str(num_dirac) + '_' + str(blur))
        f.write(' best_acc ' + str(best_criterion))
        f.write(' alpha beta ' + str(alpha) + ' ' + str(beta))
        f.write(' Epoch %i, Acc: %f ' % (checkpoint_eval['epoch'],test_acc))
        f.write('time: ' + str(run_time))
    return


for repeat in range(1):
    for alpha in alpha_list:
        for beta in beta_list:
            for test_file in test_file_list:
                print(alpha,beta, test_file)
                start_time =  time.time()
                if not os.path.exists('office_home_new_val/home_checkpoint_AED' + str(repeat)):
                    os.makedirs('office_home_new_val/home_checkpoint_AED'+ str(repeat))
                if not os.path.exists('office_home_new_val/history_home_AED'+ str(repeat)):
                    os.makedirs('office_home_new_val/history_home_AED'+ str(repeat))

                check_folder = 'office_home_new_val/home_checkpoint_AED'+ str(repeat) + '/'
                history_folder = 'office_home_new_val/history_home_AED'+ str(repeat) + '/'

                datasets = OfficeHome(test_file, test_split=0.2)
                generator = datasets.generator(test_file, batch_size=batch_size)
                test_data = datasets.getTestData(test_file)
                val_data = datasets.getValData(test_file)
                print('test data, val data', test_data[0].size(), val_data[0].size())

                clf_model = Clf(out_features=65).to(device)
                ae_model = ResNet(dropout).to(device)
                d_model = Decoder().to(device)
                optimizer_ae = optim.Adam((list(ae_model.parameters()) + list(d_model.parameters()) + list(clf_model.parameters())), lr=learning_rate)
                # optimizer_clf = optim.Adam(clf_model.parameters(), lr=learning_rate)
                AE_model, Clf_model, history_dict = train_model(ae_model, clf_model, d_model, generator, val_data,
                                                                 num_epochs, batch_size,alpha,beta,blur,num_dirac)

                history_path = history_folder + 'epoch' + str(num_epochs) + 'lr' + str(learning_rate) \
                        + 'bs' + str(batch_size)+ 'blur' + str(blur) + 'alpha' + str(alpha) + 'beta' + str(beta)

                np.save(history_path +'_history_' + test_file + '_AE.npy', history_dict)
                # save results
                check_path_eval_loss = check_folder + 'epoch' + str(num_epochs) + 'lr' + str(learning_rate) \
                        + 'bs' + str(batch_size)+ 'blur' + str(blur) + 'alpha' + str(alpha) +'beta'+ str(beta) +test_file + 'loss_model_best.pth.tar'
                end_time = time.time()
                run_time = round(end_time - start_time, 5)
                get_result(check_path_eval_loss, test_data, batch_size, run_time)
