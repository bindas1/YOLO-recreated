import torch
import wandb
from onemetric.cv.object_detection import MeanAveragePrecision
from onemetric.cv.object_detection import ConfusionMatrix
import utils
import matplotlib.pyplot as plt


SMOOTH = 1e-6

CLASSES = list({'person': 0, 'bird': 1, 'cat': 2, 'cow': 3, 'dog': 4, 'horse': 5, 'sheep': 6, 'aeroplane': 7,
                'bicycle': 8, 'boat': 9, 'bus': 10, 'car': 11, 'motorbike': 12, 'train': 13, 'bottle': 14,
                'chair': 15, 'dining table': 16, 'potted plant': 17, 'sofa': 18, 'tvmonitor': 19}.keys())

def evaluate_model(model, dataloader, config, test_dl=True, iou_threshold=0.5):
    predictions, labels = utils.pred_and_target_boxes_map(dataloader, model, single_batch=config.is_one_batch)

    mean_avg_prec = MeanAveragePrecision.from_detections(
        true_batches=labels, 
        detection_batches=predictions, 
        num_classes=config.classes, 
        iou_threshold=iou_threshold
    )

    confusion_matrix = ConfusionMatrix.from_detections(
        true_batches=labels, 
        detection_batches=predictions,
        num_classes=config.classes
    )

    dl_name = "Test set"
    if not test_dl:
        dl_name = "Train set"

    confusion_matrix.plot(".", class_names=CLASSES, normalize=True)
    plt.title("{} confusion matrix".format(dl_name))
    plt.show()

    wandb.log({"{} MAP".format(dl_name): mean_avg_prec.value})
    print("{} MAP for this model is equal to: {}".format(dl_name, mean_avg_prec.value))


def evaluate_predictions(pred_file, labels_file, is_one_batch=False):
    predictions = np.load(pred_file, allow_pickle=True).tolist()
    labels = np.load(labels_file, allow_pickle=True).tolist()

    mean_avg_prec = MeanAveragePrecision.from_detections(
        true_batches=labels, 
        detection_batches=predictions, 
        num_classes=config.classes, 
        iou_threshold=iou_threshold
    )

    confusion_matrix = ConfusionMatrix.from_detections(
        true_batches=labels, 
        detection_batches=predictions,
        num_classes=config.classes
    )

    confusion_matrix.plot(".", class_names=CLASSES, normalize=True)
    plt.title("Confusion matrix")
    plt.show()


# my own map
def mean_average_precision(predictions, targets, iou_threshold=0.5, C=20):
    ...

def mean_average_precision(predicted_boxes, target_boxes, iou_threshold=0.5, C=20):
    """
    Params:
        predicted_box: list containing all predicted bounding boxes in format
                    [[predicted_class, confidence, xmin, ymin, xmax, ymax], ...]
    """
    # false negatives - we didn't output a box for an object
    # true positives - we output a good iou box for an object
    # false positives - we output a box without good iou
    # PRECISION = TP / (TP + FP)
    # RECALL = TP / (TP + FN)

    avg_precisions = []
    last_image_index = target_boxes[-1][0]

    for class_ in range(C):
        total_tp = 0
        total_fp = 0

        class_predictions = [box for box in predicted_boxes if box[1] == class_]
        class_targets = [box for box in target_boxes if box[1] == class_]

        for i in range(last_image_index):
            c_predicted = [box[3:] for box in class_predictions if box[0]==i]
            c_targets = [box[3:] for box in class_targets if box[0]==i]
            tp_image, fp_image = metrics_for_oneimage(class_predictions, class_targets, iou_threshold)
            total_tp += tp_image
            total_fp += fp_image

        precision = total_tp / (total_fp + total_tp)
        recall = total_tp / len(class_targets)

        # add avg precision for this class...

    return avg,precisions, sum(avg_precisions) / len(avg_precisions)


def precision():
    # TP / (TP+FP)
    ...

def recall():
    # TP / (TP+FN)
    ...

def metrics_for_oneimage(class_predictions, class_targets, iou_threshold):
    true_positives = 0
    false_positives = 0
    is_predicted = [0] * len(class_targets)

    for prediction in class_predictions:
        best_iou = 0
        best_target = 0

        for index, target in enumerate(class_targets):
            iou = utils.IOU(prediction, target)
            if iou > best_iou:
                best_iou = iou
                best_target = index

        # if the iou is higher and if there isn't yet a prediction for this object
        if best_iou > iou_threshold and is_predicted[best_target]==0:
            true_positives += 1
            is_predicted[index] = 1
        else:
            false_positives += 1

    return true_positives, false_positives