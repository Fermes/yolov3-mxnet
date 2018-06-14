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
    box1[:, 0:] = np.where(box1[:, 0:] > 0., box1[:, 0:], np.zeros_like(box1[:, 0:], dtype="float32"))
    box2[:, 0:] = np.where(box2[:, 0:] > 0., box2[:, 0:], np.zeros_like(box2[:, 0:], dtype="float32"))
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
    iou = np.where(iou > 0.1, iou, np.zeros_like(iou))
    return iou


def predict_transform(prediction, inp_dim, anchors, num_classes, stride):
    ctx = prediction.context
    batch_size = prediction.shape[0]
    stride = stride
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    prediction = prediction.reshape((batch_size, bbox_attrs * num_anchors, grid_size * grid_size))
    prediction = prediction.transpose(axes=(0, 2, 1))
    prediction = prediction.reshape((batch_size, grid_size * grid_size * num_anchors, bbox_attrs))
    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]

    # Sigmoid the  centre_X, centre_Y. and object confidencce
    # mask_0 = nd.zeros(shape=prediction.shape, ctx=ctx)
    # mask_0[:, :, (0, 1, 4)] = 1
    #
    # prediction_1 = (mask_0 == 0) * prediction + (mask_0 == 1) * nd.sigmoid(prediction)

    score_pred = prediction.slice_axis(begin=4, end=5, axis=-1)
    score = nd.sigmoid(score_pred)

    # Add the center offsets
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    x_offset = nd.array(a, dtype="float32", ctx=ctx).reshape((-1, 1))
    y_offset = nd.array(b, dtype="float32", ctx=ctx).reshape((-1, 1))

    #     if CUDA:
    #         x_offset = x_offset.cuda()
    #         y_offset = y_offset.cuda()

    x_y_offset = nd.concat(x_offset, y_offset, dim=1).repeat(repeats=num_anchors, axis=1).reshape((-1, 2)).expand_dims(
        0)

    # mask_1 = nd.zeros(shape=prediction.shape, ctx=ctx)
    # mask_1[:, :, :2] = 1
    # tmp = nd.zeros(prediction.shape, ctx=ctx)
    # tmp[:, :, :2] = x_y_offset
    # prediction_2 = (mask_1 == 0) * prediction_1 + (mask_1 == 1) * (prediction_1 + tmp)
    xy_pred = prediction.slice_axis(begin=0, end=2, axis=-1)
    xy = (nd.sigmoid(xy_pred) + x_y_offset) * stride

    # log space transform height and the width
    if not isinstance(anchors, nd.NDArray):
        anchors = nd.array(anchors, dtype="float32", ctx=ctx)
    else:
        anchors = anchors.astype("float32").copyto(ctx)

    #     if CUDA:
    #         anchors = anchors.cuda()

    anchors = anchors.repeat(repeats=grid_size * grid_size, axis=0).expand_dims(0)
    wh_pred = prediction.slice_axis(begin=2, end=4, axis=-1)

    wh = nd.exp(wh_pred) * anchors * stride
    cls_pred = prediction.slice_axis(begin=5, end=None, axis=-1)
    cls = nd.sigmoid(cls_pred)
    # prediction[:, :, 2:4] = nd.exp(prediction[:, :, 2:4]) * anchors
    #
    # prediction[:, :, 5: 5 + num_classes] = nd.sigmoid((prediction[:, :, 5: 5 + num_classes]))

    # prediction[:, :, :4] *= stride
    # mask_2 = nd.zeros(shape=prediction_2.shape, ctx=ctx)
    # mask_2[:, :, 2:4] = 1
    # tmp = nd.ones(shape=prediction_2.shape, ctx=ctx)
    # tmp[:, :, 2:4] = anchors
    # prediction_3 = (mask_2 == 0) * prediction_2 + (mask_2 == 1) * (nd.exp(prediction_2) * tmp)
    # mask_3 = nd.zeros(shape=prediction_3.shape, ctx=ctx)
    # mask_3[:, :, 5: 5 + num_classes] = 1
    # prediction_4 = (mask_3 == 0) * prediction_3 + (mask_3 == 1) * nd.sigmoid(prediction_3)
    # mask_4 = nd.zeros(prediction_4.shape, ctx=ctx)
    # mask_4[:, :, :4] = 1

    return nd.concat(*[xy, wh], dim=2), score, cls


def write_results(prediction, confidence, num_classes, nms_conf=0.4):
    conf_mask = (prediction[:, :, 4] > confidence).expand_dims(2)
    for x_box in range(prediction.shape[0]):
        for y_box in range(prediction.shape[1]):
            if prediction[x_box, y_box, 4].asscalar() > confidence:
                print(prediction[x_box, y_box])
    prediction = prediction * conf_mask

    box_corner = nd.zeros(prediction.shape, dtype="float32")
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    batch_size = prediction.shape[0]

    write = False

    for ind in range(batch_size):
        image_pred = prediction[ind]
        # confidence threshholding
        # NMS
        max_conf = nd.max(image_pred[:, 5:5 + num_classes], axis=1)
        max_conf_score = nd.argmax(image_pred[:, 5:5 + num_classes], axis=1)
        max_conf = max_conf.astype("float32").expand_dims(1)
        max_conf_score = max_conf_score.astype("float32").expand_dims(1)
        # seq = (image_pred[:, :5], max_conf, max_conf_score)
        image_pred = nd.concat(image_pred[:, :5], max_conf, max_conf_score, dim=1).asnumpy()

        non_zero_ind = np.array(np.nonzero(image_pred[:, 4])).squeeze()
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
            class_mask_ind = np.squeeze(np.nonzero(cls_mask[:, -2]))
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


def prep_image(img, inp_dim, ctx=mx.cpu()):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """
    img = (letterbox_image(img, (inp_dim, inp_dim)))
    img = nd.array(img.transpose((2, 0, 1)), ctx=ctx).astype("float32")
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
