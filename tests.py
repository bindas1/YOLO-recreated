import dataset
import utils
import unittest
from PIL import Image
import torch


device = "cpu"

class TestVOCDataset(unittest.TestCase):
    def test_parse(self):
        objects = {'annotation': 
            {'folder': 'VOC2007', 'filename': '000002.jpg', 
                'source': {'database': 'The VOC2007 Database', 'annotation': 'PASCAL VOC2007', 'image': 'flickr', 'flickrid': '329145082'}, 
                'owner': {'flickrid': 'hiromori2', 'name': 'Hiroyuki Mori'}, 
                'size': {'width': '335', 'height': '500', 'depth': '3'}, 
                'segmented': '0', 'object': [{'name': 'train', 'pose': 'Unspecified', 'truncated': '0', 'difficult': '0', 'bndbox': {'xmin': '139', 'ymin': '200', 'xmax': '207', 'ymax': '301'}}]}}

        objects = objects["annotation"]["object"]
        test = dataset.VOCDataset(2007, "test")
        refer_value = [{'name': 'train', 'bndbox': {'xmin': '139', 'ymin': '200', 'xmax': '207', 'ymax': '301'}, 'label': 13}]


        self.assertEqual(test._parse_objects(objects), refer_value, "Should properly parse objects")

    def test_create_label_tensor(self):
        # TODO!!!
        pass

    # def test_resize_imaeg(self):
    #     image = np.asarray(Image.open("./image/test_image.jpg"))

    #     test = VOCDataset(2007, "test")

    #     self.assertEqual(

    #     )

class TestUtils(unittest.TestCase):
    def test_iou_for_identical_boxes(self):
        # for two identical boxes iou should be 1 for all xywh

        box_test = torch.rand(64, 7, 7, 4).to(device)
        self.assertEqual(
            torch.all(utils.IOU_tensor(box_test, box_test, device=device) == 1),
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

if __name__ == '__main__':
    unittest.main()