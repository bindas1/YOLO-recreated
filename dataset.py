import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.datasets import VOCDetection
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
import utils
import os
from albumentations.pytorch import ToTensorV2
import albumentations as A
import cv2
from tqdm import tqdm


# CONSTANTS
SIZE = 448, 448
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
classes_dict = {'person': 0, 'bird': 1, 'cat': 2, 'cow': 3, 'dog': 4, 'horse': 5, 'sheep': 6, 'aeroplane': 7,
                'bicycle': 8, 'boat': 9, 'bus': 10, 'car': 11, 'motorbike': 12, 'train': 13, 'bottle': 14,
                'chair': 15, 'diningtable': 16, 'pottedplant': 17, 'sofa': 18, 'tvmonitor': 19}
inverse_classes_dict = {v: k for k, v in classes_dict.items()}

# size of grid
S = 7
# number of bounding boxes per grid cell
B = 2
# no classes
C = 20


class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, year, image_set, include_difficult, transforms):
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

        # apply transformations, convert to tensor and normalize
        if self.transforms:
            image, objects = self._extended_transformations(image, objects, SIZE)
        else:
            # resize image and scale bounding box coordinates
            image, objects = self._default_transformations(image, objects, SIZE)

        label_tensor = self._create_label_tensor(objects, SIZE)

        return image, np.asarray(objects), label_tensor

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def collate_function(data):
        images = []
        objects_list = []

        for image, objects, _ in data:
            images.append(image)
            objects_list.append(objects)

        labels_matrix_batch = data[0][2].unsqueeze(0)
        for _, _, label_matrix in data[1:]:
            labels_matrix_batch = torch.cat([labels_matrix_batch, label_matrix.unsqueeze(0)], axis=0)

        images = torch.stack(images, dim=0)

        return images, objects_list, labels_matrix_batch

    def _remove_difficult_objects(self, objects):
        if not self.include_difficult:
            # remove only if the difficult == 1
            return [object_ for object_ in objects if object_["difficult"] != "1"]
        else:
            return objects

    @staticmethod
    def _parse_objects(objects):
        """Converts object dictionary such that it includes only 'name',
        'bdnbox' and numerical 'label' for each object and then changes it to list of bboxes"""

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
                
        # # return new objects if u want dict, else proceed further to get list
        # return new_objects
        
        # we want list [xmin, ymin, xmax, ymax, label] for transform
        bboxes = []

        for object_dict in new_objects:
            x_min = float(object_dict["bndbox"]["xmin"])
            y_min = float(object_dict["bndbox"]["ymin"])
            x_max = float(object_dict["bndbox"]["xmax"])
            y_max = float(object_dict["bndbox"]["ymax"])
            class_id = int(object_dict["label"])
            bbox = np.array([x_min, y_min, x_max, y_max, class_id])
            bboxes.append(bbox)
            
        return bboxes

    @staticmethod
    def _default_transformations(img_arr, objects, size):
        """
        :param img_arr: original image as a numpy array
        :param objects: list containing all objects that should be resized
        :param size: size of the resized image
        """
        # create resize transform pipeline that resizes to SIZE, normalizes and converts to tensor
        transform = A.Compose([
            A.Resize(height=size[1], width=size[0], always_apply=True),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc'))

        transformed = transform(image=img_arr, bboxes=objects)

        return transformed["image"], transformed["bboxes"]

    @staticmethod
    def _extended_transformations(img_arr, objects, size):
        """
        :param img_arr: original image as a numpy array
        :param objects: list containing all objects that should be resized
        :param size: size of the resized image
        """
        # create resize transform pipeline that resizes to SIZE, normalizes and converts to tensor
        transform = A.Compose([
            A.Resize(height=size[1], width=size[0], always_apply=True),
            A.Compose([
                    A.RandomCrop(width=380, height=380, p=1.0),
                    A.Resize(height=size[1], width=size[0], always_apply=True)
            ], p=0.05),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.01),
            A.Blur(blur_limit=3, p=0.5),
            A.OneOf([
                A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.6),
                A.ColorJitter(p=0.4),
            ], p=1.0),
            A.Rotate(limit=12, p=0.4, border_mode=cv2.BORDER_REFLECT_101),
            A.augmentations.transforms.Cutout(num_holes=4, max_h_size=3, max_w_size=3, p=0.1),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.5))

        transformed = transform(image=img_arr, bboxes=objects)

        return transformed["image"], transformed["bboxes"]

    @staticmethod
    def _create_label_tensor(objects, size):
        label_matrix = torch.zeros((S, S, 5 + C))

        for bbox in objects:
            class_label = int(bbox[4])
            x, y, w, h = utils.xyxy_to_xywh(
                bbox[0], bbox[1], bbox[2], bbox[3], size
            )

            # in yolo cell should be 0..1 in x and y
            i, j = int(S * y), int(S * x)
            x_relative_to_cell = S * x - j
            y_relative_to_cell = S * y - i
            # w and h should be relative to the image, so no multiplication by S
            # w_relative_to_cell = w
            # h_relative_to_cell = h

            # if there isn't an object in the cell
            if label_matrix[i, j, 0] != 1:
                label_matrix[i, j, 0] = 1
                box_coords = torch.tensor(
                    [x_relative_to_cell, y_relative_to_cell, w, h]
                )
                label_matrix[i, j, 1:5] = box_coords
                label_matrix[i, j, class_label + 5] = 1

        return label_matrix


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
        tensor_copy = tensor.clone()
        for t, m, s in zip(tensor_copy, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor_copy


# prepare the dataset
def prepare_data(batch_size, include_difficult, transforms, years, num_workers=2):
    # load dataset

    train_datasets = []

    for year in years:
        if year == 2007:
            train_datasets.append(VOCDataset(year, "train", include_difficult, transforms))
        else:
            train_datasets.append(VOCDataset(year, "trainval", include_difficult, transforms))

    # train_datasets = [
    #     VOCDataset(year, "trainval", include_difficult, transforms) for year in years
    # ]
    
    train = ConcatDataset(train_datasets)
    train_dl = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=train_datasets[0].collate_function
    )

    # val_datasets = [
    #     VOCDataset(year, "val", include_difficult, transforms=False) for year in years
    # ]
    # val = ConcatDataset(val_datasets)
    # val_dl = DataLoader(
    #     val,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=num_workers,
    #     collate_fn=val_datasets[0].collate_function
    # )

    test = VOCDataset(2007, "val", include_difficult, transforms=False)
    test_dl = DataLoader(
        test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=test.collate_function
    )

    # Check training batch
    train_features, train_objects, train_label_matrix = next(iter(train_dl))
    print(f"Feature batch shape for training: {train_features.size()}")
    print(f"Objects batch shape for training: {len(train_objects)}")
    print(f"Labels matrix batch shape for training: {train_label_matrix.size()}")

    print(f"Size of training set: {len(train_dl)*batch_size}")
    print(f"Size of validation set: {len(test_dl)*batch_size}")

    print("Sample batch for training dataloader is presented below:")
    utils.show_images_batch(train_dl, batch_size)

    return train_dl, test_dl

def prepare_test_data(batch_size, include_difficult=False, num_workers=2):
    test = VOCDataset(2007, "test", include_difficult, transforms=False)
    test_dl = DataLoader(
        test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=test.collate_function
    )

    return test_dl

def check_distribution(data_loader, save=False, title="Distrybucja.png"):
    """
    Given data_loader checks the distribution of all objects present in dataset
    """
    objects_dist, images_dist = get_distribution(data_loader)
    # get pandas dataframe with distributions
    objects_dist_df = pd.DataFrame(np.concatenate((objects_dist.reshape(-1,1), images_dist.reshape(-1,1)), axis=1), columns=["Liczba obiekt??w", "Liczba zdj???? z obiektem"]).reset_index()
    objects_dist_df = objects_dist_df.rename({'index': 'Klasa obiektu'}, axis='columns')
    objects_dist_df = objects_dist_df.replace({"Klasa obiektu": inverse_classes_dict})
    plot_distributions(objects_dist_df, save, title)

def get_distribution(data_loader):
    objects_dist = np.zeros(20)
    images_dist = np.zeros(20)

    for _, labels, _ in data_loader:
        for object_array in labels:
            classes_array = np.zeros(20)
            for box in object_array:
                class_category = int(box[4])
                objects_dist[class_category] += 1
                
                if classes_array[class_category] != 1:
                    classes_array[class_category] = 1
            images_dist += classes_array

    return objects_dist, images_dist

def plot_distributions(df, save, title):
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(20, 10), dpi=200)

    plt.subplot(1, 2, 1)
    sns.barplot(x="Liczba obiekt??w", y="Klasa obiektu", color='b', data=objects_dist_df, alpha=0.4)
    plt.title("Dystrybucja obiekt??w na wszystkich obrazkach")
    plt.grid(axis='x')

    plt.subplot(1, 2, 2)
    ax = sns.barplot(x="Liczba zdj???? z obiektem", y="Klasa obiektu", color='c', data=objects_dist_df, alpha=0.4)
    ax.set(ylabel=None)
    plt.title("Liczba zdj???? z danymi obiektami")
    plt.grid(axis='x')

    if save:
        plt.savefig(title)
    plt.show()


# prepare the dataset
def save_test(year):
    year = str(year)

    test_dataset = VOCDetection(
        root="./data",
        year=year,
        image_set="test",
        transform=None,
        download=True
    )

    cwd = os.getcwd()
    path = os.path.join(cwd, "data/test")

    for image, annotations in test_dataset:
        filename = annotations["annotation"]["filename"]
        image_path = os.path.join(path, filename)
        image.save(image_path, 'JPEG')

        basename = filename.split('.')[0]
        annotations_path = os.path.join(path, basename + ".txt")
        with open(annotations_path, 'a') as f:
            for class_object in annotations["annotation"]["object"]:
                x_min = class_object["bndbox"]["xmin"]
                y_min = class_object["bndbox"]["ymin"]
                x_max = class_object["bndbox"]["xmax"]
                y_max = class_object["bndbox"]["ymax"]  

                f.write(class_object["name"] + " " + x_min + " " + y_min + " " + x_max + " " +y_max + "\n")

        with open(os.path.join(path, "test.txt"), "a") as f:
            f.write(os.path.join(cwd, "data/test/") + filename + "\n")


# prepare the dataset
def save_test_my_dataset(folder_path, predictions):
    path = folder_path


    for idx in tqdm(range(len(predictions))):
        annotations_path = os.path.join(path, str(idx) + ".txt")

        # creates empty file
        with open(annotations_path, 'w') as fp:
            pass
        
        if predictions[idx].shape[0] == 0:
            # create empty file
            continue
        else:
            for x, y, x2, y2, class_no, conf in predictions[idx]:
                class_no = inverse_classes_dict[int(class_no)]
                x, y = str(int(x)), str(int(y))
                x2, y2 = str(int(x2)), str(int(y2))

                with open(annotations_path, 'a') as f:
                    f.write(class_no + " " + str(conf)+ " " + x + " " + y + " " + x2 + " " + y2 + "\n")


if __name__ == '__main__':
    save_test_my_dataset(2007)
    # ...

