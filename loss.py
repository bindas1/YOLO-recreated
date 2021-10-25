import utils
import torch.nn as nn
import torch


class YoloLoss(nn.Module):
    def __init__(self):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
    
    def forward(self, output, target):
        output = output.reshape(-1, S, S, C + B * 5)

        classes_probabilities_output = output[..., 10:]
        classes_probabilities_target = target[..., 5:]

        predicted_box, confidence = _get_predicted_box_and_confidence(output, target)

        # I only penalize if object exists in the cell
        exists_object_filter = (target[..., 0:1] == 1) * 1
        exists_object_filter = exists_object_filter.to(float)
        not_exists_object_filter = 1.0 - exists_object_filter
        box_targets = exists_object_filter * target[..., 1:5]

        # ==========================
        # x, y, w, h part of loss
        # ==========================
        # derivative of sqrt(0) is going to be inifinity so we add SMOOTH
        # could be negative - use sign
        predicted_box = torch.sign(predicted_box) * torch.sqrt(
            torch.abs(predicted_box + SMOOTH)
        )
        box_targets = torch.sqrt(box_targets)


        # (N, S, S, 4) -> (N*S*S, 4) -> 1
        box_loss = LAMBDA_COORD * self.mse(
            torch.flatten(predicted_box, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )

        # (N*S*S, 4) -> (N*S*S)
        loss = box_loss

        # ==========================
        # object loss (confidence)
        # ==========================
        # (N, S, S, 1) -> (N*S*S)
        object_loss = self.mse(
            torch.flatten(exists_object_filter * confidence),
            torch.flatten(exists_object_filter)
        )
        loss += object_loss

        # ==========================
        # no object loss (confidence)
        # ==========================

        no_object_loss = LAMBDA_NOOBJ * self.mse(
            torch.flatten(not_exists_object_filter * confidence),
            torch.flatten(not_exists_object_filter)
        )
        loss += no_object_loss

        # ==========================
        # class probabilities loss
        # ==========================
        # (N, S, S, 4) -> (N*S*S, 4)
        class_loss = self.mse(
            torch.flatten(exists_object_filter * classes_probabilities_output, end_dim=-2),
            torch.flatten(exists_object_filter * classes_probabilities_target, end_dim=-2)
        )

        loss += class_loss

        return loss

    def _get_predicted_box_and_confidence(self, output, target):
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



