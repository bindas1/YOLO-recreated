import matplotlib.pyplot as plt
import cv2
import numpy as np
import torchvision
from torch.utils.tensorboard import SummaryWriter
import dataset
import torch


FONT = cv2.FONT_HERSHEY_PLAIN
green = (0, 255, 0)
red = (255, 0, 0)
thickness = 1
font_size = 1.5
S = 7
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# === PLOTTING ===

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
        plt.figure(figsize=(25, 15))
        plt.imshow(np.transpose(np_img, (1, 2, 0)), aspect='auto')


# === COORDINATES CONVERSION ===

def xyxy_to_xywh(x1, y1, x2, y2, size):
    # divide by width / height to normalize to 0...1
    x = (x1 + x2) / (2 * size[0])
    y = (y1 + y2) / (2 * size[1])
    w = (x2 - x1) / size[0]
    h = (y2 - y1) / size[1]
    return x, y, w, h


def xywh_to_xyxy_pixel(x, y, w, h, size):
    x1 = (x - w / 2) * size[0]
    y1 = (y - h / 2) * size[1]
    x2 = (x + w / 2) * size[0]
    y2 = (y + h / 2) * size[1]
    return x1, y1, x2, y2

def xywh_to_xyxy(x, y, w, h, size):
    x1 = (x - w / 2)
    y1 = (y - h / 2)
    x2 = (x + w / 2)
    y2 = (y + h / 2)
    return x1, y1, x2, y2

def scale_to_image_xywh(x, y, w, h, S=S, device=device):
    # the coordinates are scaled to particular cell, so we need to add the cell number and divide by number of cells
    grid_x = torch.cat(S*[torch.Tensor(range(S)).reshape(1, -1)], axis=0).to(device)
    grid_y = torch.cat(S*[torch.Tensor(range(S)).reshape(-1, 1)], axis=1).to(device)

    # we have to multiply by the size of the grid
    x_relative_to_image = (x + grid_x) / S
    y_relative_to_image = (y + grid_y) / S
    w_relative_to_image = w
    h_relative_to_image = h
    return x_relative_to_image, y_relative_to_image, w_relative_to_image, h_relative_to_image

def xywh_to_xyxy_tensor(x, y, w, h, S=S, device=device): 
    x_image, y_image, w_image, h_image = scale_to_image_xywh(x, y, w, h, S=S, device=device)
    return xywh_to_xyxy(x_image, y_image, w_image, h_image, (S, S))


# === IOU and Non Max SUPPRESSION ===

def IOU(box_predicted, box_target):
    # I assume that the box is a list of 4 coordinates xmin, ymin, xmax, ymax
    x1_overlap = max(box_predicted[0], box_target[0])
    y1_overlap = max(box_predicted[1], box_target[1])
    x2_overlap = min(box_predicted[2], box_target[2])
    y2_overlap = min(box_predicted[3], box_target[3])

    x1_union = min(box_predicted[0], box_target[0])
    y1_union = min(box_predicted[1], box_target[1])
    x2_union = max(box_predicted[2], box_target[2])
    y2_union = max(box_predicted[3], box_target[3])

    area_overlap = (x2_overlap - x1_overlap) * (y2_overlap - y1_overlap)
    area_union = (x2_union - x1_union) * (y2_union - y1_union) - 2 * (x1_overlap - x1_union) * (y2_union - y2_overlap)

    return area_overlap / area_union

def IOU_tensor(box_predicted, box_target, device=device):
    # I assume that the box is a list of 4 coordinates xmin, ymin, xmax, ymax
    # OVERLAP
    x_pred = box_predicted[..., 0]
    y_pred = box_predicted[..., 1]
    w_pred = box_predicted[..., 2]
    h_pred = box_predicted[..., 3]

    x_target = box_predicted[..., 0]
    y_target = box_predicted[..., 1]
    w_target = box_predicted[..., 2]
    h_target = box_predicted[..., 3]

    x1_pred, y1_pred, x2_pred, y2_pred = xywh_to_xyxy_tensor(
      x_pred, y_pred, w_pred, h_pred, device=device
    )

    x1_target, y1_target, x2_target, y2_target = xywh_to_xyxy_tensor(
      x_target, y_target, w_target, h_target, device=device
    )

    x1_overlap = torch.max(x1_pred, x1_target)
    y1_overlap = torch.max(y1_pred, y1_target)
    x2_overlap = torch.min(x2_pred, x2_target)
    y2_overlap = torch.min(y2_pred, y2_target)

    x1_union = torch.min(x1_pred, x1_target)
    y1_union = torch.min(y1_pred, y1_target)
    x2_union = torch.max(x2_pred, x2_target)
    y2_union = torch.max(y2_pred, y2_target)

    area_overlap = (x2_overlap - x1_overlap) * (y2_overlap - y1_overlap)
    area_union = (x2_union - x1_union) * (y2_union - y1_union) - 2 * (x1_overlap - x1_union) * (y2_union - y2_overlap)

    return area_overlap / area_union

def non_max_suppression(predicted_boxes, iou_threshold, threshold):
    """
    Performs non max suppression on the predicted boxes


    Params:
      predicted_box: list containing all predicted bounding boxes in format
        [[predicted_class, confidence, xmin, ymin, xmax, ymax], ...]
      iou threshold: threshold to check if bounding box is correct
      threshold: threshold to check if bounding box has enough confidence of this bounding box
    """
    # filter threshold
    predicted_boxes = [bbox for bbox in predicted_boxes if bbox[1] > threshold]

    # we need to choose first the box with the highest confidence so we sort by this param
    predicted_boxes.sort(reverse=True, key=lambda b: b[1])

    nms_boxes = []

    # while there exists element in predicted_boxes
    while predicted_boxes:
        # selects and removes from list
        bbox = bboxes.pop(0)

        # remove all bboxes that are of the same class and the iou is higher than iou_threshold
        for compare_bbox in predicted_boxes:
            if bbox[0] != compare_bbox[0]:
                continue
            else:
                if IOU(bbox[2:], compare_bbox[2:]) > iou_threshold:
                    # remove compare_bbox from predicted_boxes
                    predicted_boxes.remove(compare_bbox)

        nms_boxes.append(bbox)

    return nms_boxes


# === CHECKPOINTS ===

def save_checkpoint(model, filename="yolo_checkpoint.pth.tar"):
    print("--- Saving checkpoint ---")
    torch.save(model.state_dict(), filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("--- Loading checkpoint ---")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])



