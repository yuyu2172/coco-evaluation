import os

from chainer.dataset import download

from chainercv import utils


root = 'pfnet/chainercv/coco'
img_urls = {
    'train2014': 'http://images.cocodataset.org/zips/train2014.zip',
    'val2014': 'http://images.cocodataset.org/zips/val2014.zip',
    'test2014': 'http://images.cocodataset.org/zips/test2014.zip',
    'test2015': 'http://images.cocodataset.org/zips/test2015.zip',
}
anno_urls = {
    'train2014': 'http://msvocds.blob.core.windows.net/annotations-1-0-3/'
    'instances_train-val2014.zip',
    'val2014': 'http://msvocds.blob.core.windows.net/annotations-1-0-3/'
    'instances_train-val2014.zip',
    'valminusminival2014': 'https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/'  # NOQA
    'instances_valminusminival2014.json.zip',
    'minival2014': 'https://dl.dropboxusercontent.com/s/o43o90bna78omob/'
    'instances_minival2014.json.zip',
    'test2014': 'http://images.cocodataset.org/annotations/'
    'image_info_test2014.zip',
    'test2015': 'http://images.cocodataset.org/annotations/'
    'image_info_test2015.zip',
    'test2015-dev': 'http://images.cocodataset.org/annotations/'
    'image_info_test2015.zip'
}


def get_coco(split, img_split, data_dir=None):
    url = img_urls[img_split]
    if data_dir is None:
        data_dir = download.get_dataset_directory(root)
    img_root = os.path.join(data_dir, 'images')
    created_img_root = os.path.join(img_root, img_split)
    if 'test' in split:
        annos_root = data_dir
    else:
        annos_root = os.path.join(data_dir, 'annotations')
    if 'test' in split:
        anno_prefix = 'image_info'
    else:
        anno_prefix = 'instances'
    anno_fn = os.path.join(
        annos_root, '{0}_{1}.json'.format(anno_prefix, split))
    if not os.path.exists(created_img_root):
        download_file_path = utils.cached_download(url)
        ext = os.path.splitext(url)[1]
        utils.extractall(download_file_path, img_root, ext)
    if not os.path.exists(anno_fn):
        anno_url = anno_urls[split]
        download_file_path = utils.cached_download(anno_url)
        ext = os.path.splitext(anno_url)[1]
        utils.extractall(download_file_path, annos_root, ext)
    return data_dir


coco_instance_segmentation_label_names = (
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'dining table',
    'toilet',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush')
