import os
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision.models import AlexNet
from torch.autograd import Variable
from torch.optim import lr_scheduler


import h5py
import numpy as np
from scipy.misc import imresize
from common.utils import *


class BatchImageGenerator:
    def __init__(self, flags, stage, file_path, b_unfold_label):

        if stage not in ['train', 'val', 'test']:
            assert ValueError('invalid stage!')

        self.configuration(flags, stage, file_path)
        self.load_data(b_unfold_label)

    def configuration(self, flags, stage, file_path):
        self.batch_size = flags.batch_size
        self.current_index = -1
        self.file_path = file_path
        self.stage = stage

    def normalize(self, inputs):

        # the mean and std used for the normalization of
        # the inputs for the pytorch pretrained model
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # norm to [0, 1]
        inputs = inputs / 255.0

        inputs_norm = []
        for item in inputs:
            item = np.transpose(item, (2, 0, 1))
            item_norm = []
            for c, m, s in zip(item, mean, std):
                c = np.subtract(c, m)
                c = np.divide(c, s)
                item_norm.append(c)

            item_norm = np.stack(item_norm)
            inputs_norm.append(item_norm)

        inputs_norm = np.stack(inputs_norm)

        return inputs_norm

    def load_data(self, b_unfold_label):
        file_path = self.file_path
        f = h5py.File(file_path, "r")
        self.images = np.array(f['images'])
        self.labels = np.array(f['labels'])
        f.close()

        def resize(x):
            x = x[:, :,
                  [2, 1, 0]]  # we use the pre-read hdf5 data file from the download page and need to change BRG to RGB
            return imresize(x, (224, 224, 3))

        # resize the image to 224 for the pretrained model
        self.images = np.array(list(map(resize, self.images)))

        # norm the image value
        self.images = self.normalize(self.images)

        assert np.max(self.images) < 5.0 and np.min(self.images) > -5.0

        # shift the labels to start from 0
        self.labels -= np.min(self.labels)

        if b_unfold_label:
            self.labels = unfold_label(
                labels=self.labels, classes=len(np.unique(self.labels)))
        assert len(self.images) == len(self.labels)

        self.file_num_train = len(self.labels)
        print('data num loaded:', self.file_num_train)

        if self.stage is 'train':
            self.images, self.labels = shuffle_data(
                samples=self.images, labels=self.labels)

    def get_images_labels_batch(self):

        images = []
        labels = []
        for index in range(self.batch_size):
            self.current_index += 1

            # void over flow
            if self.current_index > self.file_num_train - 1:
                self.current_index %= self.file_num_train

                self.images, self.labels = shuffle_data(
                    samples=self.images, labels=self.labels)

            images.append(self.images[self.current_index])
            labels.append(self.labels[self.current_index])

        images = np.stack(images)
        labels = np.stack(labels)

        return images, labels


def alexnet(num_classes,  pretrained=True):
    """AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet()
    if pretrained:
        model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'))
        print('Load pre trained model')
    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
    nn.init.xavier_uniform_(model.classifier[-1].weight, .1)
    nn.init.constant_(model.classifier[-1].bias, 0.)
    return model


class ModelAggregate:
    def __init__(self, flags):

        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        self.setup(flags)
        self.setup_path(flags)
        self.configure(flags)

    def setup(self, flags):
        torch.backends.cudnn.deterministic = flags.deterministic
        print('torch.backends.cudnn.deterministic:',
              torch.backends.cudnn.deterministic)
        fix_all_seed(flags.seed)

        self.network = alexnet(num_classes=flags.num_classes, pretrained=True)
        self.network = self.network.cuda()

        print(self.network)
        print('flags:', flags)
        if not os.path.exists(flags.logs):
            os.makedirs(flags.logs)

        flags_log = os.path.join(flags.logs, 'flags_log.txt')
        write_log(flags, flags_log)

        self.load_state_dict(flags, self.network)

    def setup_path(self, flags):

        root_folder = flags.data_root
        train_data = ['art_painting_train.hdf5',
                      'cartoon_train.hdf5',
                      'photo_train.hdf5',
                      'sketch_train.hdf5']

        val_data = ['art_painting_val.hdf5',
                    'cartoon_val.hdf5',
                    'photo_val.hdf5',
                    'sketch_val.hdf5']

        test_data = ['art_painting_test.hdf5',
                     'cartoon_test.hdf5',
                     'photo_test.hdf5',
                     'sketch_test.hdf5']

        self.train_paths = []
        for data in train_data:
            path = os.path.join(root_folder, data)
            self.train_paths.append(path)

        self.val_paths = []
        for data in val_data:
            path = os.path.join(root_folder, data)
            self.val_paths.append(path)

        unseen_index = flags.unseen_index

        self.unseen_data_path = os.path.join(
            root_folder, test_data[unseen_index])
        self.train_paths.remove(self.train_paths[unseen_index])
        self.val_paths.remove(self.val_paths[unseen_index])

        if not os.path.exists(flags.logs):
            os.makedirs(flags.logs)

        flags_log = os.path.join(flags.logs, 'path_log.txt')
        write_log(str(self.train_paths), flags_log)
        write_log(str(self.val_paths), flags_log)
        write_log(str(self.unseen_data_path), flags_log)

        self.batImageGenTrains = []
        for train_path in self.train_paths:
            batImageGenTrain = BatchImageGenerator(flags=flags, file_path=train_path, stage='train',
                                                   b_unfold_label=False)
            self.batImageGenTrains.append(batImageGenTrain)

        self.batImageGenVals = []
        for val_path in self.val_paths:
            batImageGenVal = BatchImageGenerator(flags=flags, file_path=val_path, stage='val',
                                                 b_unfold_label=False)
            self.batImageGenVals.append(batImageGenVal)

        self.batImageGenTest = BatchImageGenerator(flags=flags, file_path=self.unseen_data_path, stage='test',
                                                   b_unfold_label=False)

    def load_state_dict(self, flags, nn):

        if flags.state_dict:

            try:
                tmp = torch.load(flags.state_dict)
                if 'state' in tmp.keys():
                    pretrained_dict = tmp['state']
                else:
                    pretrained_dict = tmp
            except:
                pretrained_dict = model_zoo.load_url(flags.state_dict)

            model_dict = nn.state_dict()

            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                               k in model_dict and v.size() == model_dict[k].size()}

            print('model dict keys:', len(model_dict.keys()),
                  'pretrained keys:', len(pretrained_dict.keys()))
            print('model dict keys:', model_dict.keys(),
                  'pretrained keys:', pretrained_dict.keys())
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            nn.load_state_dict(model_dict)

    def configure(self, flags):

        for name, para in self.network.named_parameters():
            print(name, para.size())

        self.optimizer = sgd(parameters=self.network.parameters(),
                             lr=flags.lr,
                             weight_decay=flags.weight_decay,
                             momentum=flags.momentum)

        self.scheduler = lr_scheduler.StepLR(
            optimizer=self.optimizer, step_size=flags.step_size, gamma=0.1)
        self.loss_fn = torch.nn.cross_entropy_loss()

    def train(self, flags):
        self.network.train()
        self.network.bn_eval()
        self.best_accuracy_val = -1

        for ite in range(flags.loops_train):

            self.scheduler.step(epoch=ite)

            # get the inputs and labels from the data reader
            total_loss = 0.0
            for index in range(len(self.batImageGenTrains)):
                images_train, labels_train = self.batImageGenTrains[index].get_images_labels_batch(
                )

                inputs, labels = torch.from_numpy(
                    np.array(images_train, dtype=np.float32)), torch.from_numpy(
                    np.array(labels_train, dtype=np.float32))

                # wrap the inputs and labels in Variable
                inputs, labels = Variable(inputs, requires_grad=False).cuda(), \
                    Variable(labels, requires_grad=False).long().cuda()

                # forward with the adapted parameters
                outputs, _ = self.network(x=inputs)

                # loss
                loss = self.loss_fn(outputs, labels)

                total_loss += loss

            # init the grad to zeros first
            self.optimizer.zero_grad()

            # backward your network
            total_loss.backward()

            # optimize the parameters
            self.optimizer.step()

            if ite < 500 or ite % 500 == 0:
                print(
                    'ite:', ite, 'total loss:', total_loss.cpu().item(), 'lr:',
                    self.scheduler.get_lr()[0])

            flags_log = os.path.join(flags.logs, 'loss_log.txt')
            write_log(
                str(total_loss.item()),
                flags_log)

            if ite % flags.test_every == 0 and ite is not 0:
                self.test_workflow(self.batImageGenVals, flags, ite)

    def test_workflow(self, batImageGenVals, flags, ite):

        accuracies = []
        for count, batImageGenVal in enumerate(batImageGenVals):
            accuracy_val = self.test(batImageGenTest=batImageGenVal, flags=flags, ite=ite,
                                     log_dir=flags.logs, log_prefix='val_index_{}'.format(count))

            accuracies.append(accuracy_val)

        mean_acc = np.mean(accuracies)

        if mean_acc > self.best_accuracy_val:
            self.best_accuracy_val = mean_acc

            acc_test = self.test(batImageGenTest=self.batImageGenTest, flags=flags, ite=ite,
                                 log_dir=flags.logs, log_prefix='dg_test')

            f = open(os.path.join(flags.logs, 'Best_val.txt'), mode='a')
            f.write(
                'ite:{}, best val accuracy:{}, test accuracy:{}\n'.format(ite, self.best_accuracy_val,
                                                                          acc_test))
            f.close()

            if not os.path.exists(flags.model_path):
                os.makedirs(flags.model_path)

            outfile = os.path.join(flags.model_path, 'best_model.tar')
            torch.save(
                {'ite': ite, 'state': self.network.state_dict()}, outfile)

    def bn_process(self, flags):
        if flags.bn_eval == 1:
            self.network.bn_eval()

    def test(self, flags, ite, log_prefix, log_dir='logs/', batImageGenTest=None):

        # switch on the network test mode
        self.network.eval()

        if batImageGenTest is None:
            batImageGenTest = BatchImageGenerator(
                flags=flags, file_path='', stage='test', b_unfold_label=True)

        images_test = batImageGenTest.images
        labels_test = batImageGenTest.labels

        threshold = 50
        if len(images_test) > threshold:

            n_slices_test = int(len(images_test) / threshold)
            indices_test = []
            for per_slice in range(n_slices_test - 1):
                indices_test.append(
                    int(len(images_test) * (per_slice + 1) / n_slices_test))
            test_image_splits = np.split(
                images_test, indices_or_sections=indices_test)

            # Verify the splits are correct
            test_image_splits_2_whole = np.concatenate(test_image_splits)
            assert np.all(images_test == test_image_splits_2_whole)

            # split the test data into splits and test them one by one
            test_image_preds = []
            for test_image_split in test_image_splits:
                images_test = Variable(torch.from_numpy(
                    np.array(test_image_split, dtype=np.float32))).cuda()
                tuples = self.network(images_test)

                predictions = tuples[-1]['Predictions']
                predictions = predictions.cpu().data.numpy()
                test_image_preds.append(predictions)

            # concatenate the test predictions first
            predictions = np.concatenate(test_image_preds)
        else:
            images_test = Variable(torch.from_numpy(
                np.array(images_test, dtype=np.float32))).cuda()
            tuples = self.network(images_test)

            predictions = tuples[-1]['Predictions']
            predictions = predictions.cpu().data.numpy()

        accuracy = compute_accuracy(
            predictions=predictions, labels=labels_test)
        print('----------accuracy test----------:', accuracy)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        f = open(os.path.join(log_dir, '{}.txt'.format(log_prefix)), mode='a')
        f.write('ite:{}, accuracy:{}\n'.format(ite, accuracy))
        f.close()

        # switch on the network train mode
        self.network.train()
        self.bn_process(flags)

        return accuracy
