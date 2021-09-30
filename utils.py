import matplotlib.pyplot as plt
import cv2
import numpy as np
import torchvision
from torch.utils.tensorboard import SummaryWriter
import dataset

FONT = cv2.FONT_HERSHEY_PLAIN
green = (0, 255, 0)
red = (255, 0, 0)
thickness = 1
font_size = 1.5


def show_image_with_classes(image, labels):
    un_norm = dataset.DeNormalize(dataset.MEAN, dataset.STD)
    # denormalize the image
    npimg = un_norm(image.clone()).numpy()
    npimg = npimg.transpose((1, 2, 0)).copy()

    for object_dict in labels:
        x = int(object_dict["bndbox"]["xmin"])
        y = int(object_dict["bndbox"]["ymin"])
        x2 = int(object_dict["bndbox"]["xmax"])
        y2 = int(object_dict["bndbox"]["ymax"])
        class_name = object_dict["name"]

        cv2.rectangle(npimg, (x, y), (x2, y2), green, thickness)
        cv2.putText(npimg, class_name, (x, y), FONT, font_size, red, thickness + 2)

    # Display the image
    plt.imshow(npimg)


def show_images_batch(loader):
    writer = SummaryWriter()

    # get one batch of training images
    data_iter = iter(loader)
    images, labels, _ = data_iter.next()

    # create grid of images
    img_grid = torchvision.utils.make_grid(images)

    # show images
    matplotlib_imshow(img_grid, one_channel=False)

    # write to tensorboard
    writer.add_image('batch of VOC dataset', img_grid)


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    # class can be static prob???
    un_norm = dataset.DeNormalize(dataset.MEAN, dataset.STD)
    # denormalize the image
    img = un_norm(img)
    np_img = img.numpy()
    if one_channel:
        plt.imshow(np_img, cmap="Greys")
    else:
        plt.figure(figsize=(25, 5))
        plt.imshow(np.transpose(np_img, (1, 2, 0)), aspect='auto')


def xyxy_to_xywh(x1, y1, x2, y2, size):
    # divide by width / height to normalize to 0...1
    x = (x1 + x2) / (2 * size[0])
    y = (y1 + y2) / (2 * size[1])
    w = (x2 - x1) / size[0]
    h = (y2 - y1) / size[1]
    return x, y, w, h


def xywh_to_xyxy(x, y, w, h, size):
    x1 = (x - w / 2) * size[0]
    y1 = (y - h / 2) * size[1]
    x2 = (x + w / 2) * size[0]
    y2 = (y + h / 2) * size[1]
    return x1, y1, x2, y2
