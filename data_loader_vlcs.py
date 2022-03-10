import scipy.io as sp
import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# np.random.seed(0)
# torch.manual_seed(1)


def load_data(file_name, is_test):
    if is_test:
        data = np.load('/data/vlcs/'+file_name + '_org_data.npy')
    else:
        data = np.load('/data/vlcs/'+file_name + '_aug_data.npy')
    print(file_name, is_test)
    label = np.load('/data/vlcs/'+file_name + '_label.npy')
    return data, label

class Datasets():
    def __init__(self):
        pass

    def __load_data__(self):
        pass

    def __split_train_test__(self):
        pass


class VLCS(Datasets):
    def __init__(self, test_file, test_split=0.2):
        super(VLCS, self).__init__()
        self.data, self.label, self.source_name = self.__load_data__(test_file)
        self.test_split = test_split
        self.data_train, self.data_test, self.label_train, self.label_test = self.__split_train_test__()
        self.nDims = self.data_train[0].shape[1]
        print('name\ttrain_shape\ttest_shape\t')
        for idx, (name) in enumerate(self.source_name):
            print(name, '\t', self.data_train[idx].shape, '\t', self.data_test[idx].shape)


    def __load_data__(self, test_file):
        c_data, c_label = load_data('C', test_file=='C')
        l_data, l_label = load_data('L', test_file=='L')
        s_data, s_label = load_data('S', test_file=='S')
        v_data, v_label = load_data('V', test_file=='V')

        data = np.array([c_data, l_data, s_data, v_data])
        label = np.array([c_label, l_label, s_label, v_label])

        return data, label, ['C', 'L', 'S', 'V']

    def __split_train_test__(self):
        data_train = []
        data_test = []
        label_train = []
        label_test = []
        for i in range(len(self.data)):
            length = self.data[i].shape[0]
            test_size = int(self.test_split * length)
            keys = list(range(length))
            np.random.shuffle(keys)
            # train_idx = keys[test_size:]
            test_idx = keys[:test_size]

            # test_idx = np.random.choice(np.arange(length), test_size, replace=False)
            data_test.append(self.data[i][test_idx])
            data_train.append(np.delete(self.data[i], test_idx, axis=0))
            label_test.append(self.label[i][test_idx])
            label_train.append(np.delete(self.label[i], test_idx, axis=0))
        return np.array(data_train), np.array(data_test), np.array(label_train), np.array(label_test)

    def generator(self, testSource, batch_size=32):
        sourceId = self.source_name.index(testSource)
        trainSamples = np.delete(self.data_train, sourceId)
        trainLabels = np.delete(self.label_train, sourceId)
        trainDomainIds = [np.ones(trainLabels[i].shape) * i for i in range(len(trainLabels))]

        # batch_count = 0
        while True:
            sampleId = [np.random.choice(np.arange(len(item)), batch_size,replace=False) for item in trainLabels]
            batch_x = np.concatenate(
                [trainSamples[i][sampleId[i]] for i in range(len(sampleId))], axis=0)

            batch_y = np.concatenate(
                [trainLabels[i][sampleId[i]] for i in range(len(sampleId))], axis=0)

            batch_d = np.concatenate(
                [trainDomainIds[i][sampleId[i]] for i in range(len(sampleId))], axis=0)
            # print(np.unique(batch_d))
            batch_x = torch.from_numpy(batch_x).type('torch.FloatTensor').to(device)
            batch_y = torch.from_numpy(batch_y).type('torch.LongTensor').to(device)
            batch_d = torch.from_numpy(batch_d).type('torch.LongTensor').to(device)
            yield (batch_x,  batch_y, batch_d)

    def getValData(self, testSource):
        sourceId = self.source_name.index(testSource)
        valSamples = np.delete(self.data_test, sourceId)
        valLabels = np.delete(self.label_test, sourceId)
        s, l = np.concatenate(valSamples, axis=0), \
               np.concatenate(valLabels, axis=0).astype(int)
        index = np.arange(s.shape[0])
        np.random.shuffle(index)
        s = s[index]
        l = l[index]

        s = torch.from_numpy(s).type('torch.FloatTensor').to(device)
        l = torch.from_numpy(l).type('torch.LongTensor').to(device)
        return s, l

    def getTestData(self, testSource):
        sourceId = self.source_name.index(testSource)
        testSamples = self.data[sourceId]
        testLabels = self.label[sourceId]

        testSamples = torch.from_numpy(testSamples).type('torch.FloatTensor').to(device)
        testLabels = torch.from_numpy(testLabels).type('torch.LongTensor').to(device)
        return testSamples, testLabels
