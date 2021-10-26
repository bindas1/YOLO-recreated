import utils
import torch.nn as nn
import torch
from dataset import S, C, B


# from the paper
MOMENTUM = 0.9
EPOCHS = 10
DECAY = 0.0005
# model is trained with 135 epochs
# first 5 epochs from 0.001 to 0.1
# 75 epochs 0.01 epochs
# 30 epochs 0.001
# 30 epochs 0.0001
LEARNING_RATE = 0.0001
LAMBDA_NOOBJ = 0.5
LAMBDA_COORD = 5
SMOOTH = 1e-6


class YoloLoss(nn.Module):
    def __init__(self):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
    
    def forward(self, output, target):
        output = output.reshape(-1, S, S, C + B * 5)

        predicted_box, confidence = self._get_predicted_box_and_confidence(output, target)

        # I only want to penalize if object exists in the cell
        exists_object_filter = (target[..., 0:1] == 1) * 1
        exists_object_filter = exists_object_filter.to(float)
    
        loss = self._xywh_loss(predicted_box, target[..., 1:5], exists_object_filter)
        loss += self._object_loss(confidence, exists_object_filter)
        loss += self._no_object_loss(confidence, exists_object_filter)
        loss += self._class_loss(output, target, exists_object_filter)

        return loss

    @staticmethod
    def _get_predicted_box_and_confidence(output, target):
        """
        Calculates IOU for both boxes and for each sample chooses the coordinates of the better boxes
        Calculates also the confidence of the correct predictions
        """
        # calculate iou for both boxes
        iou_box1 = utils.IOU_tensor(output[..., 1:5], target[..., 1:5])
        iou_box2 = utils.IOU_tensor(output[..., 6:10], target[..., 1:5])

        # filters to see which bounding box is better
        # >= because if we multiply by 1 False becomes 0
        better_box1 = (iou_box1 >= iou_box2) * 1
        better_box1.to(float)
        better_box2 = 1.0 - better_box1

        # for each image get better box from cell, add dimension to better_box
        predicted_box = better_box1[..., None] * output[..., 1:5] + better_box2[..., None] * output[..., 6:10]
        predicted_box = predicted_box.to(float)
        confidence = better_box1[..., None] * output[..., 0:1] + better_box2[..., None] * output[..., 5:6]
        confidence = confidence.to(float)

        return predicted_box, confidence

    def _xywh_loss(self, predicted_box, box_targets, exists_object_filter):
        # ==========================
        # x, y, w, h part of loss
        # ==========================

        predicted_box = exists_object_filter * predicted_box
        box_targets = exists_object_filter * box_targets

        # derivative of sqrt(0) is going to be inifinity so we add SMOOTH
        # could be negative - use sign
        predicted_box = exists_object_filter * torch.sign(predicted_box) * torch.sqrt(
            torch.abs(predicted_box + SMOOTH)
        )
        box_targets = torch.sqrt(box_targets)

        # (N, S, S, 4) -> (N*S*S, 4) -> 1
        box_loss = LAMBDA_COORD * self.mse(
            torch.flatten(predicted_box, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )

        return box_loss

    def _object_loss(self, confidence, exists_object_filter):
        # ==========================
        # object loss (confidence)
        # ==========================

        # exists_object_filter already has target[..., 0:1] included so not needed Confidence_target
        # (N, S, S, 1) -> (N*S*S)
        object_loss = self.mse(
            torch.flatten(exists_object_filter * confidence),
            torch.flatten(exists_object_filter)
        )

        return object_loss

    def _no_object_loss(self, confidence, exists_object_filter):
        # ==========================
        # no object loss (confidence)
        # ==========================

        not_exists_object_filter = 1.0 - exists_object_filter

        no_object_loss = LAMBDA_NOOBJ * self.mse(
            torch.flatten(not_exists_object_filter * confidence),
            torch.flatten(not_exists_object_filter)
        )

        return no_object_loss

    def _class_loss(self, output, target, exists_object_filter):
        # ==========================
        # class probabilities loss
        # ==========================

        classes_probabilities_output = output[..., 10:]
        classes_probabilities_target = target[..., 5:]

        # (N, S, S, 4) -> (N*S*S, 4)
        class_loss = self.mse(
            torch.flatten(exists_object_filter * classes_probabilities_output, end_dim=-2),
            torch.flatten(exists_object_filter * classes_probabilities_target, end_dim=-2)
        )

        return class_loss


