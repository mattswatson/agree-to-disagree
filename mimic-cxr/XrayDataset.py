import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from PIL import Image

import os
import h5py

class XrayDataset(Dataset):
    def __init__(self, base_path, split='train', label='Pneumonia', ignore_empty_labels=False, return_dicom_path=False,
                 transform=None, verbose=False):
        super(XrayDataset, self).__init__()

        self.base_path = base_path
        self.img_path = os.path.join(self.base_path, 'files')
        self.split = split
        self.label = label
        self.ignore_empty_labels = ignore_empty_labels
        self.return_dicom_path = return_dicom_path
        self.transforms = transform
        self.verbose = verbose

        if self.split not in ['train', 'test', 'validate', 'whole']:
            raise Exception('Dataset split must be one of train, test, validate or whole')

        # The base path must include the metadata files
        path = os.path.join(self.base_path, 'mimic-cxr-2.0.0-metadata.csv')
        try:
            self.dicom_metadata_df = pd.read_csv(path)
        except FileNotFoundError:
            raise Exception("CXR metadata file {} does not exist".format(path))

        path = os.path.join(self.base_path, 'mimic-cxr-2.0.0-split.csv')
        try:
            self.split_df = pd.read_csv(path)
        except FileNotFoundError:
            raise Exception("Data split file {} does not exist".format(path))

        path = os.path.join(self.base_path, 'mimic-cxr-2.0.0-chexpert.csv')
        try:
            self.labels_df = pd.read_csv(path)
        except FileNotFoundError:
            raise Exception("Labels file {} does not exist".format(path))

        # This is a multi-label dataset (i.e. one CXR can have multiple labels)
        # If we are looking for a specific label, we will return 0 or 1:
        # 0 - The label was negatively mentioned (i.e. the disease is not present)
        # 1 - The label was positively mentioned (i.e. the disease is present)
        # If ignore_empty_labels is False, then we assume a label of 0 if the label is not mentioned in the report
        # (Note that other papers have set a precedent for this, so we have citations for the method)
        # If ignore_empty_labels is True, then we ignore all CXRs where our desired label has not been mentioned

        # First, get a list of the possible labels
        possible_labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum',
                           'Fracture', 'Lung Lesion', 'Lung Opacity', 'Pleural Effusion', 'Pneumonia', 'Pneumothorax',
                           'Pleural Other', 'Support Devices', 'No Finding']

        if self.label not in possible_labels:
            raise Exception('Label {} does not exist in dataset'.format(self.label))

        # Get all of the records in our dataset split
        if self.split != 'whole':
            records = self.split_df[self.split_df['split'] == self.split]
            print('size of split {} is {}'.format(self.split, len(records)))
        else:
            records = self.split_df

        dicoms_with_positive_label = self.labels_df[self.labels_df[self.label] == 1.0]
        if self.ignore_empty_labels:
            dicoms_with_negative_label = self.labels_df[self.labels_df[self.label] == 0.0]
        else:
            print('Using values with no label as negative')
            dicoms_with_negative_label = pd.concat((self.labels_df[self.labels_df[self.label] == 0.0],
                                                    self.labels_df[self.labels_df[self.label].isnull()]))

            print('overall num with -ve/no label:', len(dicoms_with_negative_label))

        records_with_positive_label = records[records['study_id'].isin(dicoms_with_positive_label['study_id'])]
        records_with_negative_label = records[records['study_id'].isin(dicoms_with_negative_label['study_id'])]

        print('Num positive labels:', len(records_with_positive_label))
        print('Num negative labels:', len(records_with_negative_label))

        rows = []
        for index, record in records_with_positive_label.iterrows():
            row = [record['dicom_id'], record['study_id'], record['subject_id'], 1]
            rows.append(row)

        for index, record in records_with_negative_label.iterrows():
            row = [record['dicom_id'], record['study_id'], record['subject_id'], 0]
            rows.append(row)

        self.records = pd.DataFrame(rows, columns=['dicom_id', 'study_id', 'subject_id', 'label'])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        record = self.records.iloc[item]
        image_path = self.get_cxr_path(str(record['dicom_id']), str(record['study_id']), str(record['subject_id']))
        cxr = Image.open(image_path).convert('RGB')

        if self.verbose:
            print('image path:', image_path)

        if self.transforms is None:
            composed = transforms.Compose([transforms.ToTensor()])
        else:
            composed = self.transforms

        cxr = composed(cxr)

        if self.return_dicom_path:
            return cxr, torch.tensor(record['label']), image_path

        return cxr, torch.tensor(record['label'])

    def get_cxr_path(self, dicom_id, study_id, subject_id):
        # Get the first three digits of the patient id
        parent_folder = str(subject_id)[:2]

        cxr_path = os.path.join(self.base_path, 'files', 'p{}'.format(parent_folder), 'p{}'.format(subject_id),
                                's{}'.format(study_id), '{}.jpg'.format(dicom_id))
        return cxr_path


class XrayDatasetHDF5(XrayDataset):
    def __init__(self, base_path, hdf5_path, split='train', label='Pneumonia', ignore_empty_labels=False,
                 return_dicom_path=False):
        super(XrayDatasetHDF5, self).__init__(base_path, split, label, ignore_empty_labels, return_dicom_path)

        self.hdf5_path = hdf5_path
        self.cxr_images = h5py.File(self.hdf5_path, 'r')

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        record = self.records.iloc[item]
        image_path = XrayDataset.get_cxr_path(self, str(record['dicom_id']), str(record['study_id']),
                                              str(record['subject_id']))
        image_path = os.path.expanduser(image_path)

        print('image_path:', image_path)
        print(self.cxr_images.keys())
        cxr_group = self.cxr_images.get(image_path)
        print('group:', cxr_group)
        cxr = torch.tensor(np.array(cxr_group.get('img')))

        if self.return_dicom_path:
            return cxr, torch.tensor(record['label']), image_path

        return cxr, torch.tensor(record['label'])

if __name__ == '__main__':
    dataset = XrayDataset('/home2/gskj82/data/mimic-cxr/physionet.org/files/mimic-cxr-jpg/2.0.0/',
                          split='train', label='Edema', ignore_empty_labels=True)
    test_img = dataset[0][0]

    from torchvision.utils import save_image
    save_image(test_img, './test_cxr_img.png')
