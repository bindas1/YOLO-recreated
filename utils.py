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
SMOOTH = 1e-6

classes_dict = {'person': 0, 'bird': 1, 'cat': 2, 'cow': 3, 'dog': 4, 'horse': 5, 'sheep': 6, 'aeroplane': 7,
                'bicycle': 8, 'boat': 9, 'bus': 10, 'car': 11, 'motorbike': 12, 'train': 13, 'bottle': 14,
                'chair': 15, 'dining table': 16, 'potted plant': 17, 'sofa': 18, 'tvmonitor': 19}
inverse_classes_dict = {v: k for k, v in classes_dict.items()}


# === PLOTTING ===

def show_image_with_classes_(image, labels):
    un_norm = dataset.DeNormalize(dataset.MEAN, dataset.STD)
    # denormalize the image
    npimg = un_norm(image.clone()).numpy()
    npimg = npimg.transpose((1, 2, 0)).copy()

    for bbox in labels:
        x = int(bbox[0])
        y = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        class_name = inverse_classes_dict[int(bbox[4])]

        cv2.rectangle(npimg, (x, y), (x2, y2), green, thickness)
        cv2.putText(npimg, class_name, (x, y), FONT, font_size, red, thickness + 2)

    # Display the image
    plt.imshow(npimg)


def show_images_batch(loader, batch_size):
    writer = SummaryWriter()

    # get one batch of training images
    data_iter = iter(loader)
    images, labels, _ = data_iter.next()

    # create grid of images
    img_grid = torchvision.utils.make_grid(images)

    # show images
    matplotlib_imshow(img_grid, batch_size, one_channel=False)

    # write to tensorboard
    writer.add_image('batch of VOC dataset', img_grid)


def matplotlib_imshow(img, batch_size, one_channel=False):
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
        plt.figure(figsize=(batch_size, batch_size*batch_size/8/8))
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

def xywh_to_xyxy_cell(x, y, w, h, size):
    x1 = (x - w / 2 * size[0])
    y1 = (y - h / 2 * size[1])
    x2 = (x + w / 2 * size[0])
    y2 = (y + h / 2 * size[1])
    return x1, y1, x2, y2

def xywh_to_xyxy_image(x, y, w, h, size):
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
    return xywh_to_xyxy_image(x_image, y_image, w_image, h_image, (S, S))

def tensor_to_bbox_list(tensor, is_target, S=S):
    """Input is an output/target tensor of size (batch_size, S,S,30)
    is_target tells us if this is of shape (batch_size, S,S,25) if true
    """
    if not is_target:
        # first find out which box is better
        confidence_for_box1 = tensor[..., 0]
        confidence_for_box2 = tensor[..., 4]

        # find better box when it comes to model confidence
        better_box1 = (confidence_for_box1 >= confidence_for_box2) * 1
        better_box1.to(float)
        better_box2 = 1.0 - better_box1

        bboxes = better_box1[..., None] * tensor[..., 1:5] + better_box2[..., None] * tensor[..., 6:10]
        bboxes.to(float)



        confidence = better_box1[..., None] * tensor[..., 0:1] + better_box2[..., None] * tensor[..., 5:6]
        confidence = confidence.to(float)
        class_prediction = tensor[..., 10:].argmax(-1, keepdim=True)
    else:
        bboxes = tensor[..., 1:5]
        confidence = tensor[..., 0:1]
        class_prediction = tensor[..., 5:].argmax(-1, keepdim=True)

    # convert from local cell coords to global image coords
    xmin, ymin, xmax, ymax = xywh_to_xyxy_tensor(
      bboxes[..., 0], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]
    )

    coords = torch.stack([xmin, ymin, xmax, ymax], -1)
    

    # resize to (batch_size, S, S, 6)
    predictions = torch.cat([class_prediction, confidence, coords], dim=-1)

    # now convert tensors of (batch_size, S, S, 6) -> [[[predicted_class, confidence, xmin, ymin, xmax, ymax], ...], ...]
    # size of the list will be (batch_size, S*S, 6)
    
    predictions = predictions.reshape(predictions.size()[0], S * S, -1)

    return predictions.tolist()


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

    area_overlap = max(0, (x2_overlap - x1_overlap)) * max(0, (y2_overlap - y1_overlap))
    area_union = (x2_union - x1_union) * (y2_union - y1_union) - 2 * (x1_overlap - x1_union) * (y2_union - y2_overlap)
    
    # make sure union doesn't contain 0
    area_union += SMOOTH

    return area_overlap / area_union

def IOU_tensor(box_predicted, box_target, device=device):
    # I assume that the box is a list of 4 coordinates xmin, ymin, xmax, ymax
    # OVERLAP
    x_pred = box_predicted[..., 0]
    y_pred = box_predicted[..., 1]
    w_pred = box_predicted[..., 2]
    h_pred = box_predicted[..., 3]

    x_target = box_target[..., 0]
    y_target = box_target[..., 1]
    w_target = box_target[..., 2]
    h_target = box_target[..., 3]

    x1_pred, y1_pred, x2_pred, y2_pred = xywh_to_xyxy_cell(
      x_pred, y_pred, w_pred, h_pred, (S,S)
    )

    x1_target, y1_target, x2_target, y2_target = xywh_to_xyxy_cell(
      x_target, y_target, w_target, h_target, (S,S)
    )

    x1_overlap = torch.max(x1_pred, x1_target)
    y1_overlap = torch.max(y1_pred, y1_target)
    x2_overlap = torch.min(x2_pred, x2_target)
    y2_overlap = torch.min(y2_pred, y2_target)

    x1_union = torch.min(x1_pred, x1_target)
    y1_union = torch.min(y1_pred, y1_target)
    x2_union = torch.max(x2_pred, x2_target)
    y2_union = torch.max(y2_pred, y2_target)

    area_overlap = (x2_overlap - x1_overlap).clamp(0) * (y2_overlap - y1_overlap).clamp(0)
    area_union = (x2_union - x1_union) * (y2_union - y1_union) - 2 * (x1_overlap - x1_union) * (y2_union - y2_overlap)

    # make sure union doesn't contain 0
    area_union += SMOOTH

    iou = area_overlap / area_union
    # return torch.unsqueeze(iou, -1)
    return iou

def non_max_suppression(predicted_boxes, iou_threshold, conf_threshold):
    """
    Performs non max suppression on the predicted boxes


    Params:
      predicted_box: list containing all predicted bounding boxes in format
        [[predicted_class, confidence, xmin, ymin, xmax, ymax], ...]
      iou threshold: threshold to check if bounding box is correct
      conf_threshold: threshold to check if bounding box has enough confidence for this bounding box
    """
    # filter threshold
    predicted_boxes = [bbox for bbox in predicted_boxes if bbox[1] > conf_threshold]

    # we need to choose first the box with the highest confidence so we sort by this param
    predicted_boxes.sort(reverse=True, key=lambda b: b[1])

    nms_boxes = []

    # while there exists element in predicted_boxes
    while predicted_boxes:
        # selects and removes from list
        bbox = predicted_boxes.pop(0)

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

def pred_and_target_boxes_map(data_loader, model, iou_threshold=0.5, conf_threshold=0.2, single_batch=False):
    """Function used to obtain prediction and target boxes for evaluation and depicting results
    Results:
    - target: list of numpy arras (each numpy array for each photo) with bboxes [xmin, ymin, xmax,  ymax, label]
    - predicted_boxes: list of numpy arras (each numpy array for each photo) with bboxes [xmin, ymin, xmax,  ymax, label, confidence]
    """

    # switch to evaluation mode
    model.eval()

    predicted_boxes = []

    # index to track picture id
    pic_index = 0

    
    labels = []

    for inputs, batch_labels, _ in data_loader:
        inputs = inputs.to(device)
        batch_labels = batch_labels[:]

        # deactivate autograd -> reduce memory usage and speed up computations
        with torch.no_grad():
            # predictions are tensor (batch_size, 7, 7, 30) when S=7
            predictions = model(inputs)

        batch_size = inputs.size()[0]

        pred_bbox = tensor_to_bbox_list(predictions, is_target=False)

        for i in range(batch_size):
            nms_pred_boxes = non_max_suppression(
                pred_bbox[i], iou_threshold, conf_threshold
            )

            image_predictions = np.array([])

            for box in nms_pred_boxes:
                box = np.array(box)
                box = box[[2, 3, 4, 5, 0, 1]]
                box[:4] = box[:4] * 448
                # concatenate with rest of the labels
                if image_predictions.size:
                    image_predictions = np.vstack([image_predictions, box])
                else:
                    image_predictions = box
                    
            image_predictions = image_predictions.reshape(-1, 6)

            predicted_boxes.append(image_predictions)

            pic_index += 1

        labels += [l.reshape(-1,5) for l in batch_labels]

        if single_batch:
            break

    model.train()

    return predicted_boxes, labels

def pred_and_target_boxes(data_loader, model, single_batch=False, iou_threshold=0.5, conf_threshold=0.2):
    """Function used to obtain prediction and target boxes for evaluation and depicting results
    Results:
    - target: array of numpy arras with bboxes [photo_id, class, xmin, ymin, xmax,  ymax]
    - predicted_boxes: array of numpy arras (each numpy array for each photo) with bboxes [photo_id, class, confidence, xmin, ymin, xmax,  ymax]
    """

    # switch to evaluation mode
    model.eval()

    predicted_boxes = []
    target_boxes = []

    # index to track picture id
    pic_index = 0

    if not single_batch:
        labels = np.array([])

        for inputs, batch_labels, _ in data_loader:
            inputs = inputs.to(device)
            batch_labels = batch_labels[:]

            # deactivate autograd -> reduce memory usage and speed up computations
            with torch.no_grad():
                # predictions are tensor (batch_size, 7, 7, 30) when S=7
                predictions = model(inputs)

            batch_size = inputs.size()[0]

            pred_bbox = tensor_to_bbox_list(predictions, is_target=False)

            # get indices of empty arrays
            to_remove = []

            for i in range(batch_size):
                nms_pred_boxes = non_max_suppression(
                    pred_bbox[i], iou_threshold, conf_threshold
                )

                for box in nms_pred_boxes:
                    # add the pic index to all elements and append to predicted_boxes
                    predicted_boxes.append([pic_index] + box)

                #if the array isn't empty
                if batch_labels[i].size != 0:
                    # insert index of photo to target labels
                    batch_labels[i] = np.insert(batch_labels[i], 0, pic_index, axis=1)
                else:
                    to_remove.append(i)

                pic_index += 1

            for idx in sorted(to_remove, reverse=True):
                del batch_labels[idx]
        
            # stack labels to one big numpy array
            batch_labels = np.concatenate(batch_labels)

            # concatenate with rest of the labels
            if labels.size:
                labels = np.vstack([labels, batch_labels])
            else:
                labels = batch_labels

    else:
    # FOR ONE BATCH TO BE DELETED LATER
        inputs = data_loader[0].to(device)
        batch_labels = data_loader[1][:]

        # deactivate autograd -> reduce memory usage and speed up computations
        with torch.no_grad():
            # predictions are tensor (batch_size, 7, 7, 30) when S=7
            predictions = model(inputs)

        batch_size = inputs.size()[0]

        pred_bbox = tensor_to_bbox_list(predictions, is_target=False)

        for i in range(batch_size):
            nms_pred_boxes = non_max_suppression(
                pred_bbox[i], iou_threshold, conf_threshold
            )

            for box in nms_pred_boxes:
                # add the pic index to all elements and append to predicted_boxes
                predicted_boxes.append([pic_index] + box)
            
            # insert index of photo to target labels
            batch_labels[i] = np.insert(batch_labels[i], 0, pic_index, axis=1)

            pic_index += 1

        # stack labels to one big numpy array
        labels = np.concatenate(batch_labels)

    # change the order from [idx,x,y,x,y,c_id]->[idx,c_id,x,y,x,y]
    labels = labels[:, [0, 5, 1, 2, 3, 4]]

    model.train()

    return np.array(predicted_boxes), labels

# === CHECKPOINTS ===

def save_checkpoint(model, optimizer, filename="yolo_checkpoint.pth.tar"):
    print("--- Saving checkpoint ---")
    torch.save({
            # 'epoch': EPOCH,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
    }, filename)

def load_checkpoint(checkpoint_file, model, optimizer):
    print("--- Loading checkpoint ---")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


# ==== IF USING DICT INSTEAD OF LIST FOR LABELS

def show_image_with_classes_dict(image, labels):
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


