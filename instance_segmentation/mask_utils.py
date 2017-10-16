import numpy as np


def mask2whole_mask(mask, bbox, size):
    """Convert list representation of instance masks into an image-sized array.

    Args:
        mask (list): [(H_1, W_1), ..., (H_R, W_R)]
        bbox (array): Array of shape (R, 4)
        size (tuple of ints): (H, W)

    Returns:
        array of shape (R, H, W)

    """
    if len(mask) != len(bbox):
        raise ValueError('The length of mask and bbox should be the same')
    R = len(mask)
    H, W = size
    whole_mask = np.zeros((R, H, W), dtype=np.bool)

    for i, (m, bb) in enumerate(zip(mask, bbox)):
        bb = np.round(bb).astype(np.int32)
        whole_mask[i, bb[0]:bb[2], bb[1]:bb[3]] = m.copy()
    return whole_mask


def whole_mask2mask(whole_mask, bbox):
    """Convert an image-sized array of instance masks into a list.

    Args:
        whole_mask (array): array of shape (R, H, W)
        bbox (array): Array of shape (R, 4)

    Returns:
        [(H_1, W_1), ..., (H_R, W_R)]

    """
    mask = list()
    for i, (whole_m, bb) in enumerate(zip(whole_mask, bbox)):
        bb = np.round(bb).astype(np.int32)
        mask.append(whole_m[bb[0]:bb[2], bb[1]:bb[3]])
    return mask


if __name__ == '__main__':
    from fcis.datasets.coco_instance_segmentation_dataset import COCOInstanceSegmentationDataset
    from fcis.datasets.coco_utils import coco_instance_segmentation_label_names
    from fcis.utils import visualize_mask
    import matplotlib.pyplot as plt
    dataset = COCOInstanceSegmentationDataset(split='val')
    img, bbox, label, mask = dataset[0]
    whole_mask = mask2whole_mask(mask, bbox, img.shape[1:])
    mask = whole_mask2mask(whole_mask, bbox)
    mask = [m.astype(np.float32) for m in mask]


    img = img.transpose(1, 2, 0)
    visualize_mask(img, label, mask, bbox, np.ones(len(bbox)), coco_instance_segmentation_label_names, 0.5)
    plt.show()

    plt.imshow(np.max(whole_mask, axis=0))
    plt.show()
