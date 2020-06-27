import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from  copy import deepcopy
from collections import OrderedDict
import os

import matplotlib.image as mpimg
import pickle

def DATA_ANALYSIS():


    train_files = os.listdir('train/')
    test_files = os.listdir('test/')

    print(f"Train-set Size: {len(train_files)}")
    print(f"Test-set Size: {len(test_files)}")

    # ======================== Images demonstration ====

    images_num = 16
    fig, axs = plt.subplots(4, 4)
    for i in range(images_num):
        image = mpimg.imread(f"train/{train_files[i]}")
        axs[int(i/4), i%4].imshow(image)
        plt.xlabel(f"{train_files[i]}")
        axs[int(i / 4), i % 4].title.set_text(f"{train_files[i]}")
        axs[int(i / 4), i % 4].axes.get_xaxis().set_visible(False)
        axs[int(i / 4), i % 4].axes.get_yaxis().set_visible(False)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=1.5, wspace=None, hspace=None)
    plt.show()

    # ======================== Data insights =======
    #
    masked_num = 0
    unmasked_num = 0
    for image_name in train_files:
        if("_1." in image_name):
            masked_num +=1
        elif("_0." in image_name):
            unmasked_num +=1
        else:
            image = mpimg.imread(f"train/{image_name}")
            plt.imshow(image)
    plt.show()

    print(f"Proper mask wearers in train dataset: {masked_num}")
    print(f"Unproper mask wearers in train dataset: {unmasked_num}")

    masked_num = 0
    unmasked_num = 0
    for image_name in test_files:
        if("_1." in image_name):
            masked_num +=1
        elif("_0." in image_name):
            unmasked_num +=1
        else:
            image = mpimg.imread(f"test/{image_name}")
            plt.imshow(image)
    plt.show()

    print(f"Proper mask wearers in test dataset: {masked_num}")
    print(f"Unproper mask wearers in test dataset: {unmasked_num}")

# ======================== DATA EXCTRACTION ===============

def DF_creation(path_to_folder='train', load_save = True):
    """
    :param path_to_folder: 'train' or 'test'
    :param load_save: if True load save (if exists), else creates files from scratch
    :return:
    """
    if(path_to_folder[-1]== '/'):
        path_to_folder = path_to_folder[:-1]
    images_list = os.listdir(path_to_folder + '/')
    df_file = ''
    for el in path_to_folder.split('/'):
        df_file += el + '_'
    df_file +=  'df.pkl'
    if ((df_file in os.listdir('saves/')) and load_save):
        df = pickle.load(open(f'saves/{df_file}', "rb"))
    else:
        df = pd.DataFrame()
        for image_name in images_list:
            mask = 0
            if ("_1." in image_name):
                mask = 1
            df = df.append({
                'image': f'{path_to_folder}/' + image_name,
                'mask': mask
            }, ignore_index=True)
        pickle.dump(df, open(f'saves/{df_file}', "wb"))
    path_to_pkl = f'saves/{df_file}'
    return df, path_to_pkl

# train_df, path_to_train_pkl = DF_creation('train')
# print(train_df.head())
# test_df, path_to_test_pkl = DF_creation('test')
# print(test_df.head())

def tmp_DF_creation(path_to_folder='train', load_save = True):
    """
    :param path_to_folder: 'train' or 'test'
    :param load_save: if True load save (if exists), else creates files from scratch
    :return:
    """
    if(path_to_folder[-1]== '/'):
        path_to_folder = path_to_folder[:-1]
    images_list = os.listdir(path_to_folder + '/')
    df_file = ''
    for el in path_to_folder.split('/'):
        df_file += el + '_'
    df_file +=  'df.pkl'
    if ((df_file in os.listdir('tmp_saves/')) and load_save):
        df = pickle.load(open(f'tmp_saves/{df_file}', "rb"))
    else:
        df = pd.DataFrame()
        for image_name in images_list:
            mask = 0
            if ("_1." in image_name):
                mask = 1
            df = df.append({
                'image': f'{path_to_folder}/' + image_name,
                'mask': mask
            }, ignore_index=True)
        pickle.dump(df, open(f'tmp_saves/{df_file}', "wb"))
    path_to_pkl = f'tmp_saves/{df_file}'
    return df, path_to_pkl

# ========================= Results Data

train_results = pd.DataFrame(columns=['iteration','loss', 'ROC_AUC', 'f1_score'])
test_results = pd.DataFrame(columns=['iteration','loss', 'ROC_AUC', 'f1_score'])
prediction_df = pd.DataFrame(columns=['image','real_mask', 'predicted_mask'])

# ======================
import cv2
from torch import long, tensor
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor

class MaskDataset(Dataset):
    """ Masked faces dataset
        0 = 'not correct mask wearing'
        1 = 'correct mask wearing'
    """

    def __init__(self, dataFrame):
        self.dataFrame = dataFrame

        self.transformations = Compose([
            ToPILImage(),
            Resize((100, 100)),
            ToTensor(),  # [0, 1]
        ])

    def __getitem__(self, key):
        if isinstance(key, slice):
            raise NotImplementedError('slicing is not supported')

        row = self.dataFrame.iloc[key]
        return {
            'name': row['image'],
            'image': self.transformations(cv2.imread(row['image'])),
            'mask': tensor([row['mask']], dtype=long),  # pylint: disable=not-callable
        }

    def __len__(self):
        return len(self.dataFrame.index)

# ===============
""" Training module
"""
from pathlib import Path
from typing import Dict, List, Union

from PIL import Image
import pytorch_lightning as pl
import torch
import torch.nn.init as init
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.nn import (Conv2d, CrossEntropyLoss, Linear, MaxPool2d, ReLU,
                      Sequential)
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader


class MaskDetector(pl.LightningModule):
    """ MaskDetector PyTorch Lightning class
    """

    def __init__(self, train_df_pkl_path, test_df_pkl_path = None, epoch = 1, only_testing = False):
        """

        :param train_df_pkl_path: dumped pickle of train_df
        :param test_df_pkl_path:  dumped pickle of test_Df
        :param epoch: current epoch
        :param only_testing: True if we already pretrained model
        """
        super(MaskDetector, self).__init__()
        self.epoch = epoch
        self.trainDFPath = train_df_pkl_path
        self.testDFPath = test_df_pkl_path
        self.maskDF = None
        self.trainDF = None
        self.validateDF = None
        self.testDF = None
        self.crossEntropyLoss = None
        self.learningRate = 0.00001

        self.only_testing = only_testing

        self.convLayer1 = convLayer1 = Sequential(
            Conv2d(3, 32, kernel_size=(3, 3), padding=(1, 1)),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2))
        )

        self.convLayer2 = convLayer2 = Sequential(
            Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2))
        )

        self.convLayer3 = convLayer3 = Sequential(
            Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1), stride=(3, 3)),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2))
        )

        self.linearLayers = linearLayers = Sequential(
            Linear(in_features=2048, out_features=1024),
            ReLU(),
            Linear(in_features=1024, out_features=2),
        )

        # Initialize layers' weights
        for sequential in [convLayer1, convLayer2, convLayer3, linearLayers]:
            for layer in sequential.children():
                if isinstance(layer, (Linear, Conv2d)):
                    init.xavier_uniform_(layer.weight)

    def forward(self, x: Tensor):  # pylint: disable=arguments-differ
        """ forward pass
        """
        out = self.convLayer1(x)
        out = self.convLayer2(out)
        out = self.convLayer3(out)
        out = out.view(-1, 2048)
        out = self.linearLayers(out)
        return out

    def _set_test_dataset(self, testDFpath):
        test = pd.read_pickle(self.testDFPath)
        self.testDF = MaskDataset(test)

    def prepare_data(self) -> None:
        self.maskDF = maskDF = pd.read_pickle(self.trainDFPath)
        train, validate = train_test_split(maskDF, test_size=0.3, random_state=0,
                                           stratify=maskDF['mask'])
        self.trainDF = MaskDataset(train)
        self.validateDF = MaskDataset(validate)
        if(self.testDFPath != None):
            test = pd.read_pickle(self.testDFPath)
            self.testDF = MaskDataset(test)
        else:
            print('EMPTY TEST SET')
            self.testDF = MaskDataset(pd.DataFrame(columns=['name', 'image', 'mask']))
        # Create weight vector for CrossEntropyLoss
        maskNum = maskDF[maskDF['mask'] == 1].shape[0]
        nonMaskNum = maskDF[maskDF['mask'] == 0].shape[0]
        nSamples = [nonMaskNum, maskNum]
        normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
        self.crossEntropyLoss = CrossEntropyLoss(weight=torch.tensor(normedWeights))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.trainDF, batch_size=32, shuffle=True, num_workers=4)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.validateDF, batch_size=32, num_workers=4)

    def test_dataloader(self) -> DataLoader:
        print('TEST DATA LOADED')
        return DataLoader(self.testDF, batch_size=32, shuffle=False, num_workers=4)

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), lr=self.learningRate)

    def training_step(self, batch: dict, _batch_idx: int) -> Dict[str, Tensor]:  # pylint: disable=arguments-differ
        inputs, labels = batch['image'], batch['mask']
        labels = labels.flatten()
        outputs = self.forward(inputs)
        loss = self.crossEntropyLoss(outputs, labels)

        tensorboardLogs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboardLogs}

    def validation_step(self, batch: dict, _batch_idx: int) -> Dict[str, Tensor]:  # pylint: disable=arguments-differ
        inputs, labels = batch['image'], batch['mask']
        labels = labels.flatten()
        outputs = self.forward(inputs)
        loss = self.crossEntropyLoss(outputs, labels)

        _, outputs = torch.max(outputs, dim=1)
        valAcc = accuracy_score(outputs.cpu(), labels.cpu())
        valF1 = f1_score(outputs.cpu(), labels.cpu())
        valAUC = roc_auc_score(labels.cpu(), outputs.cpu())

        valAcc = torch.tensor(valAcc)
        valF1 = torch.tensor(valF1)
        valAUC = torch.tensor(valAUC)

        return {'val_loss': loss, 'val_acc': valAcc, 'val_f1': valF1, 'val_auc': valAUC}

    def validation_epoch_end(self, outputs: List[Dict[str, Tensor]]) \
            -> Dict[str, Union[Tensor, Dict[str, Tensor]]]:

        global train_results

        avgLoss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avgAcc = torch.stack([x['val_acc'] for x in outputs]).mean()
        avgF1 = torch.stack([x['val_f1'] for x in outputs]).mean()
        avgAUC = torch.stack([x['val_auc'] for x in outputs]).mean()

        results = {'iteration':self.epoch, 'loss':avgLoss.item(), 'ROC_AUC':avgAUC.item(), 'f1_score':avgF1.item()}
        train_results = train_results.append(results, ignore_index = True)

        tensorboardLogs = {'val_loss': avgLoss, 'val_acc': avgAcc, 'val_f1': avgF1, 'val_auc': avgAUC}
        return {'val_loss': avgLoss, 'log': tensorboardLogs}

    def test_step(self, batch: dict, _batch_idx: int) -> Dict[str, Tensor]:  # pylint: disable=arguments-differ
        global prediction_df

        inputs, labels, names = batch['image'], batch['mask'], batch['name']
        labels = labels.flatten()
        outputs = self.forward(inputs)
        loss = self.crossEntropyLoss(outputs, labels)

        _, outputs = torch.max(outputs, dim=1)
        names, real, pred = [el.split('/')[-1] for el in names], labels, outputs
        data = []
        for i, el in enumerate(names):
            data.append([names[i], real[i].item(), pred[i].item()])

        batch_df = pd.DataFrame(columns=['image','real_mask', 'predicted_mask'], data=data)
        prediction_df = prediction_df.append(batch_df, ignore_index=True)

        valAcc = accuracy_score(outputs.cpu(), labels.cpu())
        valF1 = f1_score(outputs.cpu(), labels.cpu())
        valAUC = roc_auc_score(labels.cpu(), outputs.cpu())

        valAcc = torch.tensor(valAcc)
        valF1 = torch.tensor(valF1)
        valAUC = torch.tensor(valAUC)


        return {'val_loss': loss, 'val_acc': valAcc, 'val_f1':valF1, 'val_auc': valAUC}

    def test_epoch_end(self, outputs: Union[List[Dict[str, Tensor]], List[List[Dict[str, Tensor]]]]):



        avgLoss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avgAcc = torch.stack([x['val_acc'] for x in outputs]).mean()
        avgF1 = torch.stack([x['val_f1'] for x in outputs]).mean()
        avgAUC = torch.stack([x['val_auc'] for x in outputs]).mean()

        if(self.only_testing):
            global test_results
            results = {'iteration':self.epoch, 'loss':avgLoss.item(), 'ROC_AUC':avgAUC.item(), 'f1_score':avgF1.item()}
            test_results = test_results.append(results, ignore_index = True)

        self.epoch += 1

        tensorboardLogs = {'val_loss': avgLoss, 'val_acc': avgAcc, 'val_f1':avgF1,'val_auc': avgAUC}
        return {'val_loss': avgLoss, 'log': tensorboardLogs}

def train(EPOCHS = 20):
    """

    :param EPOCHS: number of tested epochs
    :return:
    """
    global train_results
    model = MaskDetector(Path('saves/train_df.pkl'))

    checkpoint_callback = ModelCheckpoint(
        save_weights_only=True,
        save_top_k=EPOCHS,
        verbose=True,
        monitor='val_acc',
        mode='max'
    )

    trainer = Trainer(gpus=1 if torch.cuda.is_available() else 0,
                      max_epochs=EPOCHS,
                      checkpoint_callback=checkpoint_callback,
                      default_root_dir='checkpoints/',
                      row_log_interval=40,
                      resume_from_checkpoint='best_models/lightning_best_2.ckpt',
                      profiler=True)

    trainer.fit(model)
    train_results.to_csv('results/train_results.csv', header=True, index=False)

def test(path_to_test_folder='test', delete_pkl = False):
    global prediction_df

    if (path_to_test_folder[-1] == '/'):
        path_to_test_folder = path_to_test_folder[:-1]

    pkl_name = ''
    for el in path_to_test_folder.split('/'):
        pkl_name += el + '_'
    pkl_name += 'df.pkl'

    if(f'{pkl_name}' in os.listdir('saves/')):
        path_to_test_pkl = f'saves/{pkl_name}'
        model = MaskDetector(Path('saves/train_df.pkl'), Path(path_to_test_pkl), epoch = 0, only_testing=True)
    else:
        test_df, path_to_test_pkl = DF_creation(path_to_test_folder, load_save = False)
        model = MaskDetector(Path('saves/train_df.pkl'), Path(path_to_test_pkl), epoch=0, only_testing=True)

    checkpoint_callback = ModelCheckpoint(
        # filepath='checkpoints/_ckpt_epoch_20.ckpt',
        save_weights_only=True,
        save_top_k=0,
        verbose=True,
        monitor='val_acc',
        mode='max'
    )
    trainer = Trainer(gpusSize=1 if torch.cuda.is_available() else 0,
                      max_epochs=0,
                      checkpoint_callback=checkpoint_callback,
                      profiler=True,
                      default_root_dir='checkpoints/',
                      resume_from_checkpoint='best_models/lightning_best_2.ckpt',
                      row_log_interval=40
                      )
    trainer.fit(model)
    # from torchsummary import summary
    # summary(model, (3, 100, 100))
    trainer.test()

    prediction_df = prediction_df.drop('real_mask', axis=1)
    # prediction_df.to_csv('predictions.csv', header=False, index=False)
    prediction_df.to_csv('prediction.csv', header=True, index=False)
    if(delete_pkl):
        os.remove(path_to_test_pkl)

def show_test_results(EPOCHS = 20, checkpoints_directory='checkpoints/ordered_checkpoints/'):
    global test_results
    model = MaskDetector(Path('saves/train_df.pkl'), Path('saves/test_df.pkl'), only_testing=True)
    checkpoint_callback = ModelCheckpoint(
        # filepath='checkpoints/_ckpt_epoch_20.ckpt',
        save_weights_only=True,
        save_top_k=0,
        verbose=True,
        monitor='val_acc',
        mode='max'
    )
    for i in range(EPOCHS):
        trainer = Trainer(gpusSize=1 if torch.cuda.is_available() else 0,
                          max_epochs=0,
                          checkpoint_callback=checkpoint_callback,
                          profiler=True,
                          default_root_dir='checkpoints/',
                          resume_from_checkpoint=f'{checkpoints_directory}/epoch={i}.ckpt',
                          row_log_interval=40
                          )
        trainer.fit(model)
        trainer.test()
    test_results.to_csv('results/test_results.csv', header=True, index=False)
    print(f'RESULTS: {test_results}')

def results_graphs():
    train_df = pd.read_csv('graph_data/train_results.csv')
    test_df = pd.read_csv('graph_data/test_results.csv')
    # LOSS:
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    train_loss = sns.lineplot(x=range(len(train_df)), y='loss', data=train_df, label="Train")
    test_loss = sns.lineplot(x=range(len(test_df)), y='loss', data=test_df, label="Test")
    plt.show()
    # AUC:
    train_auc = sns.lineplot(x=range(len(train_df)), y='ROC_AUC', data=train_df, label="Train")
    test_auc = sns.lineplot(x=range(len(test_df)), y='ROC_AUC', data=test_df, label="Test")
    plt.show()
    # F1-score:
    train_auc = sns.lineplot(x=range(len(train_df)), y='f1_score', data=train_df, label="Train")
    test_auc = sns.lineplot(x=range(len(test_df)), y='f1_score', data=test_df, label="Test")
    plt.show()

def prediction_test(path_to_test_folder='test', delete_pkl = True):
    global prediction_df

    if (path_to_test_folder[-1] == '/'):
        path_to_test_folder = path_to_test_folder[:-1]

    pkl_name = ''
    for el in path_to_test_folder.split('/'):
        pkl_name += el + '_'
    pkl_name += 'df.pkl'

    if(f'{pkl_name}' in os.listdir('tmp_saves/')):
        path_to_test_pkl = f'tmp_saves/{pkl_name}'
        model = MaskDetector(Path('saves/train_df.pkl'), Path(path_to_test_pkl), epoch = 0, only_testing=True)
    else:
        test_df, path_to_test_pkl = tmp_DF_creation(path_to_test_folder, load_save = False)
        model = MaskDetector(Path('saves/train_df.pkl'), Path(path_to_test_pkl), epoch=0, only_testing=True)

    checkpoint_callback = ModelCheckpoint(
        # filepath='checkpoints/_ckpt_epoch_20.ckpt',
        save_weights_only=True,
        save_top_k=0,
        verbose=True,
        monitor='val_acc',
        mode='max'
    )
    trainer = Trainer(gpusSize=1 if torch.cuda.is_available() else 0,
                      max_epochs=0,
                      checkpoint_callback=checkpoint_callback,
                      profiler=True,
                      default_root_dir='checkpoints/',
                      resume_from_checkpoint='best_models/lightning_best_2.ckpt',
                      row_log_interval=40
                      )
    trainer.fit(model)
    trainer.test()

    prediction_df = prediction_df.drop('real_mask', axis=1)

    if(delete_pkl):
        os.remove(path_to_test_pkl)

    result = prediction_df.copy()
    result.rename(columns={'image': 'id', 'predicted_mask':'label'},inplace=True)
    return result

if __name__ == '__main__':
    # pass
    train(60)
    # test()
    # show_test_results(40, 'checkpoints/lightning_logs/version_56/checkpoints/')
    # results_graphs()