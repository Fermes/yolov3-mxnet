from __future__ import division
import mxnet as mx
import cv2
import numpy as np
from mxnet import nd


def bbox_iou(box1, box2, mode="xywh"):
    """
    Returns the IoU of two bounding boxes
    """
    if mode == "xywh":
        tmp_box1 = box1.copy()
        tmp_box1[:, 0] = box1[:, 0] - box1[:, 2] / 2.0
        tmp_box1[:, 1] = box1[:, 1] - box1[:, 3] / 2.0
        tmp_box1[:, 2] = box1[:, 0] + box1[:, 2] / 2.0
        tmp_box1[:, 3] = box1[:, 1] + box1[:, 3] / 2.0
        box1 = tmp_box1
        tmp_box2 = box2.copy()
        tmp_box2[:, 0] = box2[:, 0] - box2[:, 2] / 2.0
        tmp_box2[:, 1] = box2[:, 1] - box2[:, 3] / 2.0
        tmp_box2[:, 2] = box2[:, 0] + box2[:, 2] / 2.0
        tmp_box2[:, 3] = box2[:, 1] + box2[:, 3] / 2.0
        box2 = tmp_box2
    box1[box1 < 0.] = 0.
    box2[box2 < 0.] = 0.
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = np.where(b1_x1 > b2_x1, b1_x1, b2_x1)
    inter_rect_y1 = np.where(b1_y1 > b2_y1, b1_y1, b2_y1)
    inter_rect_x2 = np.where(b1_x2 < b2_x2, b1_x2, b2_x2)
    inter_rect_y2 = np.where(b1_y2 < b2_y2, b1_y2, b2_y2)

    # Intersection area
    inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, a_min=0, a_max=None) * np.clip(
        inter_rect_y2 - inter_rect_y1 + 1,
        a_min=0, a_max=None)
    inter_area[inter_area < 5] = 0.

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    iou = inter_area / (b1_area + b2_area - inter_area)
    iou[iou < 0] = 0.
    return iou


def train_transform(prediction, num_classes, stride):
    batch_size = prediction.shape[0]
    bbox_attrs = 5 + num_classes

    prediction_1 = nd.transpose(prediction.reshape((batch_size, bbox_attrs * 3, stride * stride)), (0, 2, 1))\
        .reshape(batch_size, stride * stride * 3, bbox_attrs)

    xy_pred = nd.sigmoid(prediction_1.slice_axis(begin=0, end=2, axis=-1))
    wh_pred = prediction_1.slice_axis(begin=2, end=4, axis=-1)
    score_pred = nd.sigmoid(prediction_1.slice_axis(begin=4, end=5, axis=-1))
    cls_pred = nd.sigmoid(prediction_1.slice_axis(begin=5, end=None, axis=-1))

    return nd.concat(xy_pred, wh_pred, score_pred, cls_pred, dim=-1)


def predict_transform(prediction, anchors, num_classes, stride, confidence=0.5):
    ctx = prediction.context
    batch_size = prediction.shape[0]
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    item_index = np.argwhere(prediction[:, :, 4].asnumpy() > confidence)
    if not isinstance(anchors, nd.NDArray):
        anchors = nd.array(anchors)
    grid = np.arange(stride)
    a, b = np.meshgrid(grid, grid)

    x_offset = a.reshape((-1, 1))
    y_offset = b.reshape((-1, 1))
    x_y_offset = np.repeat(np.concatenate((x_offset, y_offset), 1), repeats=num_anchors, axis=1)
    x_y_offset = nd.array(x_y_offset.reshape(-1, 2)).expand_dims(0)
    prediction[:, :, :2] += x_y_offset

    anchors = anchors.repeat(repeats=stride * stride, axis=0).expand_dims(0)
    prediction[:, :, :2] = prediction[:, :, :2] / stride * 416.0
    prediction[:, :, 2:4] = nd.exp(prediction[:, :, 2:4]) * anchors
    # for x_box, y_box in item_index:
    #     prediction[x_box, y_box, :2] = prediction[x_box, y_box, :2] / stride * 416.0
    #     prediction[x_box, y_box, 2:4] = nd.exp(prediction[x_box, y_box, 2:4]) \
    #                                     * anchors[y_box // (stride * stride)]

    return prediction


def write_results(prediction, num_classes, confidence=0.5, nms_conf=0.4):
    conf_mask = (prediction[:, :, 4] > confidence).expand_dims(2)
    prediction = prediction * conf_mask

    batch_size = prediction.shape[0]
    write = False
    box_corner = nd.zeros(prediction.shape, dtype="float32")
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    for ind in range(batch_size):
        image_pred = prediction[ind]
        # confidence threshholding
        # NMS

        max_conf = nd.max(image_pred[:, 5:5 + num_classes], axis=1)
        max_conf_score = nd.argmax(image_pred[:, 5:5 + num_classes], axis=1)
        max_conf = max_conf.astype("float32").expand_dims(1)
        max_conf_score = max_conf_score.astype("float32").expand_dims(1)
        image_pred = nd.concat(image_pred[:, :5], max_conf, max_conf_score, dim=1).asnumpy()
        non_zero_ind = np.nonzero(image_pred[:, 4])
        try:
            image_pred_ = image_pred[non_zero_ind, :].reshape((-1, 7))
        except Exception as e:
            print(e)
            continue
        if image_pred_.shape[0] == 0:
            continue
        # Get the various classes detected in the image
        img_classes = np.unique(image_pred_[:, -1])
        # -1 index holds the class index

        for cls in img_classes:
            # get the detections with one particular class
            cls_mask = image_pred_ * np.expand_dims(image_pred_[:, -1] == cls, axis=1)
            class_mask_ind = np.nonzero(cls_mask[:, -2])
            image_pred_class = image_pred_[class_mask_ind].reshape((-1, 7))

            # sort the detections such that the entry with the maximum objectness
            # confidence is at the top
            conf_sort_index = np.argsort(image_pred_class[:, 4])[::-1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.shape[0]

            for i in range(idx):
                # Get the IOUs of all boxes that come after the one we are looking at
                # in the loop
                try:
                    ious = bbox_iou(np.expand_dims(image_pred_class[i], 0), image_pred_class[i + 1:])
                except ValueError:
                    break
                except IndexError:
                    break

                # Zero out all the detections that have IoU > treshhold
                iou_mask = np.expand_dims(ious < nms_conf, 1).astype(np.float32)
                image_pred_class[i + 1:] *= iou_mask

                # Remove the non-zero entries
                non_zero_ind = np.squeeze(np.nonzero(image_pred_class[:, 4]))
                image_pred_class = image_pred_class[non_zero_ind].reshape((-1, 7))

            batch_ind = np.ones((image_pred_class.shape[0], 1)) * ind

            seq = nd.concat(nd.array(batch_ind), nd.array(image_pred_class), dim=1)

            if not write:
                output = seq
                write = True
            else:
                output = nd.concat(output, seq, dim=0)

    try:
        return output
    except:
        return 0


def letterbox_image(img, inp_dim):
    # resize image with unchanged aspect ratio using padding
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image

    return canvas


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """
    img = (letterbox_image(img, (inp_dim, inp_dim)))
    img = np.array(img.transpose((2, 0, 1))).astype("float32")
    img /= 255.0
    return img


def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.readlines()
    names = [name.strip() for name in names]
    if names[-1] == "\n":
        names.pop(-1)
    return names


def split_and_load(data, ctx):
    n, k = data.shape[0], len(ctx)
    m = n // k
    return [data[i * m: (i + 1) * m].as_in_context(ctx[i]) for i in range(k)]
