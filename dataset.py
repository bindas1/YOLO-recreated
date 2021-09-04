import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import matplotlib.patches as patches
from PIL import Image
import torch
import torchvision
from torchvision.datasets import VOCDetection
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import albumentations
import utils


# CONSTANTS
SIZE = 448, 448
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
classes_dict = {'person':0, 'bird':1, 'cat':2, 'cow':3, 'dog':4, 'horse':5, 'sheep':6, 'aeroplane':7,
                'bicycle':8, 'boat':9, 'bus':10, 'car':11, 'motorbike':12, 'train':13, 'bottle':14,
                'chair':15, 'dining table':16, 'potted plant':17, 'sofa':18, 'tvmonitor':19}


class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, year, image_set, include_difficult=False, transforms=None):
        self.image_set = image_set
        assert self.image_set in ["train", "val", "trainval", "test"]

        self.year = str(year)
        assert self.year in ["2007", "2012"]

        self.include_difficult = include_difficult
        self.transforms = transforms
        self.dataset = VOCDetection(
            root="./data",
            year=self.year,
            image_set=self.image_set,
            transform=None,
            download=True
        )

    def __getitem__(self, idx):
        image, annotations = self.dataset[idx]

        # convert image to numpy array
        image = np.asarray(image)

        # filter annotations to include only necessary data
        objects = annotations["annotation"]["object"]
        objects = self._remove_difficult_objects(objects)
        objects = self._parse_objects(objects)

        # resize image and scale bounding box coordinates
        image, objects = self._resize_image(image, objects, SIZE)

        # apply transformations, convert to tensor and normalize
        if self.transforms is not None:
            image, objects = self.transforms(image, objects)
        image = torchvision.transforms.functional.to_tensor(image)

        # Normalize by mean and standard deviation of ImageNet data that our base VGG was trained on
        image = torchvision.transforms.functional.normalize(image, mean=MEAN, std=STD)

        return image, objects

    def __len__(self):
        return len(self.dataset)

    def collate_function(self, data):
        images = []
        objects_list = []

        for image, objects in data:
            images.append(image)
            objects_list.append(objects)

        images = torch.stack(images, dim=0)

        return images, objects_list

    def _remove_difficult_objects(self, objects):
        if not self.include_difficult:
            # remove only if the difficult == 1
            return [object_ for object_ in objects if object_["difficult"] != "1"]
        else:
            return objects

    def _parse_objects(self, objects):
        """Converts object dictionary such that it includes only 'name',
        'bdnbox' and numerical 'label' for each object"""

        remove_features = ["truncated", "pose", "difficult"]
        new_objects = []

        for index, object_dict in enumerate(objects):
            # add numerical label
            if object_dict["name"] in classes_dict.keys():
                object_dict["label"] = classes_dict[object_dict["name"]]
                for key in remove_features:
                    object_dict.pop(key, None)
                new_objects.append(object_dict)
            else:
                # if it contains object other than specified in classes_dict
                # we don't want to remove it from labels
                continue
        return new_objects

    def _resize_image(self, img_arr, objects, size):
        """
        :param img_arr: original image as a numpy array
        :param h: resized height dimension of image
        :param w: resized weight dimension of image
        """
        # create resize transform pipeline
        transform = albumentations.Compose(
            [albumentations.Resize(height=size[1], width=size[0], always_apply=True)],
            bbox_params=albumentations.BboxParams(format='pascal_voc'))

        bboxes = []

        for object_dict in objects:
            x_min = int(object_dict["bndbox"]["xmin"])
            y_min = int(object_dict["bndbox"]["ymin"])
            x_max = int(object_dict["bndbox"]["xmax"])
            y_max = int(object_dict["bndbox"]["ymax"])
            class_id = int(object_dict["label"])
            bbox = np.array([x_min, y_min, x_max, y_max, class_id])
            bboxes.append(bbox)

        transformed = transform(image=img_arr, bboxes=bboxes)
        image = transformed["image"]
        bboxes = transformed["bboxes"]

        for i, object_dict in enumerate(objects):
            new_bndbox = {
                "xmin": bboxes[i][0],
                "ymin": bboxes[i][1],
                "xmax": bboxes[i][2],
                "ymax": bboxes[i][3]
            }
            object_dict["bndbox"] = new_bndbox

        return image, objects


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


# prepare the dataset
def prepare_data():
    # load dataset
    train = VOCDataset(2007, "trainval")
    test = VOCDataset(2007, "test")
    # prepare data loaders
    train_dl = DataLoader(
        train,
        batch_size=64,
        shuffle=False,
        num_workers=2,
        collate_fn=train.collate_function
    )
    test_dl = DataLoader(
        test,
        batch_size=64,
        shuffle=False,
        num_workers=2,
        collate_fn=test.collate_function
    )

    # Check training batch
    train_features, train_labels = next(iter(train_dl))
    print(f"Feature batch shape for training: {train_features.size()}")
    print(f"Labels batch shape for training: {len(train_labels)}")

    print("Sample batch for training dataloader is presented below:")
    utils.show_images_batch(train_dl)

    return train_dl, test_dl



