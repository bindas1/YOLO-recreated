import dataset
import utils
import unittest
from PIL import Image
import torch


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
    def test_identical_boxes(self):
        # for two identical boxes iou should be 1 for all xywh

        device = "cpu"
        box_test = torch.rand(64, 7, 7, 4).to(device)
        self.assertEqual(
            torch.all(utils.IOU_tensor(box_test, box_test, device=device) == 1),
            True
        )


if __name__ == '__main__':
    unittest.main()