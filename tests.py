import dataset
import utils
import unittest
from PIL import Image
import torch
import loss
import architecture
import numpy as np
import time
from tqdm.notebook import tqdm


device = "cpu"

class SpeedTesting():
    def __init__(self, model, dataset, device=None, max_iter=None):
        self.model = model
        self.dataset = dataset
        self.max_iter = -1 if max_iter is None else max_iter
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

    def speed_testing(self):
        # cuDnn configurations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

        model = self.model.to(self.device)

        model.eval()
        time_list = []

        for i in tqdm(range(len(self.dataset))):
            inputs, _, _ = self.dataset[i]
            inputs = inputs.to(self.device)
            inputs = inputs.reshape(-1, 3, 448, 448)

            torch.cuda.synchronize()
            tic = time.time()

            with torch.no_grad():
                # predictions are tensor (batch_size, 7, 7, 30) when S=7
                predictions = model(inputs)
                predictions = predictions.reshape(-1, 7, 7, 30)

            pred_bbox = utils.tensor_to_bbox_list(predictions, is_target=False, device=self.device)
            _ = utils.non_max_suppression(
                pred_bbox[0], 0.5, 0.2
            )

            time_list.append(time.time()-tic)

            if i == (self.max_iter - 1):
                break

        time_list = time_list[1:]
        print("     + Total number of iterations: {}s".format(len(time_list)))
        print("     + Total time cost: {}s".format(sum(time_list)))
        print("     + Average time cost: {}s".format(sum(time_list)/(len(time_list))))
        print("     + Frame Per Second: {:.2f}".format(len(time_list)/(sum(time_list))))    


class TestVOCDataset(unittest.TestCase):
    def test_parse(self):
        objects = {'annotation': 
            {'folder': 'VOC2007', 'filename': '000002.jpg', 
                'source': {'database': 'The VOC2007 Database', 'annotation': 'PASCAL VOC2007', 'image': 'flickr', 'flickrid': '329145082'}, 
                'owner': {'flickrid': 'hiromori2', 'name': 'Hiroyuki Mori'}, 
                'size': {'width': '335', 'height': '500', 'depth': '3'}, 
                'segmented': '0', 'object': [{'name': 'train', 'pose': 'Unspecified', 'truncated': '0', 'difficult': '0', 'bndbox': {'xmin': '139', 'ymin': '200', 'xmax': '207', 'ymax': '301'}}]}}

        objects = objects["annotation"]["object"]
        test = dataset.VOCDataset(2007, "test", False, None)

        # refer_value = [{'name': 'train', 'bndbox': {'xmin': '139', 'ymin': '200', 'xmax': '207', 'ymax': '301'}, 'label': 13}]
        refer_value = [np.array([139., 200., 207., 301.,  13.])]
        print(test._parse_objects(objects))

        self.assertEqual(test._parse_objects(objects), refer_value, "Should properly parse objects")

class TestUtils(unittest.TestCase):
    def test_iou_for_identical_boxes(self):
        # for two identical boxes iou should be 1 for all xywh

        box_test = torch.rand(64, 7, 7, 4).to(device)
        self.assertEqual(
            # all should be 1 (or very close to one)
            torch.all(utils.IOU_tensor(box_test, box_test, device=device) > 0.999),
            True
        )

    def test_scale_to_image_xywh(self):
        # this converts to global image coordinates
        x = torch.tensor([[[[0.5], [0.25]], [[0.75], [0]]],
                [[[0.5], [0.25]], [[0.75], [0]]]])[..., 0].to(device)
        y = torch.tensor([[[[0.5], [0.25]], [[0.75], [0]]],
                        [[[0.5], [0.25]], [[0.75], [0]]]])[..., 0].to(device)
        w = torch.tensor([[[[0.25], [0.125]], [[0.5], [0.5]]],
                        [[[0.25], [0.125]], [[0.5], [0.5]]]])[..., 0].to(device)
        h = torch.tensor([[[[0.5], [0.125]], [[0.25], [0.5]]],
                        [[[0.5], [0.125]], [[0.25], [0.5]]]])[..., 0].to(device)

        refer_value = (torch.tensor([[[0.2500, 0.6250],
                              [0.3750, 0.5000]],
                     
                             [[0.2500, 0.6250],
                              [0.3750, 0.5000]]], device=device),
                     torch.tensor([[[0.2500, 0.1250],
                              [0.8750, 0.5000]],
                     
                             [[0.2500, 0.1250],
                              [0.8750, 0.5000]]], device=device),
                     torch.tensor([[[0.2500, 0.1250],
                              [0.5000, 0.5000]],
                     
                             [[0.2500, 0.1250],
                              [0.5000, 0.5000]]], device=device),
                     torch.tensor([[[0.5000, 0.1250],
                              [0.2500, 0.5000]],
                     
                             [[0.5000, 0.1250],
                              [0.2500, 0.5000]]], device=device))

        self.assertEqual(
            [torch.equal(coord, refer_value) for coord, refer_value in zip(
                utils.scale_to_image_xywh(x, y, w, h, S=2, device=device), refer_value)],
            [True, True, True, True]
        )

class TestArchitecture(unittest.TestCase):
    def test_output_shape(self):
        yolo_net = architecture.darknet()
        input = torch.rand(4, 3, 448, 448)

        self.assertEqual(yolo_net(input).size(), torch.Size([4, 7, 7, 30]))


if __name__ == '__main__':
    unittest.main()