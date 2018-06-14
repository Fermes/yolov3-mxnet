import re
import threading
import time
import xml.etree.ElementTree as ET
from random import shuffle
from mxnet import autograd
from mxnet import gluon
from darknet_mxnet import DarkNet
from util_mxnet import *


def preprocess_true_boxes(label_file, input_shape, anchors, num_classes):
    '''Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    '''
    with open(label_file, "r") as file:
        true_boxes = file.readlines()
    true_boxes = np.array([list(map(float, x.split())) for x in true_boxes], dtype="float32")
    true_boxes = true_boxes[:, :, [1, 2, 3, 4, 0]]
    assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors) // 3  # default setting
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]

    input_shape = np.array(input_shape, dtype='int32')
    boxes_wh = true_boxes[..., 2:4]

    m = true_boxes.shape[0]
    grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
    y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + num_classes),
                       dtype='float32') for l in range(num_layers)]

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0

    for b in range(m):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0:
            continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b, t, 4].astype('int32')
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5 + c] = 1

    return y_true


def parse_xml(xml_file, classes):
    root = ET.parse(xml_file).getroot()
    label = {}
    image_size = root.find("size")
    size = {
        "width": float(image_size.find("width").text),
        "height": float(image_size.find("height").text),
        "depth": float(image_size.find("depth").text)
    }
    label["size"] = size
    bbox = []
    if not isinstance(classes, np.ndarray):
        classes = np.array(classes)
    for obj in root.findall("object"):
        cls = (classes == obj.find("name").text).astype(np.float)
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
        center_x = (xmin + xmax) / 2.0 / size["width"]
        center_y = (ymin + ymax) / 2.0 / size["height"]
        width = (xmax - xmin) / size["width"]
        height = (ymax - ymin) / size["height"]
        bbox.append(np.hstack(([center_x, center_y, width, height, 1.0], cls)))
    label["bbox"] = np.array(bbox)
    return np.array(bbox)


def prep_label(label_file, num_classes, ctx):
    with open(label_file, "r") as file:
        labels = file.readlines()
        labels = np.array([list(map(float, x.split())) for x in labels], dtype="float32")
        final_labels = nd.zeros(shape=(30, num_classes + 5), dtype="float32", ctx=ctx)
        i = 0
        for label in labels:
            one_hot = np.zeros(shape=(num_classes + 5), dtype="float32")
            one_hot[5 + int(label[0])] = 1.0
            one_hot[4] = 1.0
            one_hot[:4] = label[1:] * 416.0
            final_labels[i] = one_hot
            i += 1
            i %= 30
        return nd.array(final_labels, ctx=ctx)


class YoloDataSet(gluon.data.Dataset):
    def __init__(self, images_file, classes, input_dim=416, is_shuffle=False, ctx=mx.cpu()):
        super(YoloDataSet, self).__init__()
        self.anchors = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                        (59, 119), (116, 90), (156, 198), (373, 326)]
        self.classes = classes
        self.input_dim = input_dim
        self.ctx = ctx
        with open(images_file, "r") as file:
            self.image_list = file.readlines()
        self.image_list = [im.strip() for im in self.image_list]
        if is_shuffle:
            shuffle(self.image_list)
        pattern = re.compile("(\.png|\.jpg|\.bmp|\.jpeg)")
        self.label_list = []
        for i in range(len(self.image_list) - 1, -1, -1):
            if pattern.search(self.image_list[i]) is None:
                self.image_list.pop(i)
                continue
            label = pattern.sub(lambda s: ".txt", self.image_list[i]).replace("JPEGImages", "labels")
            # if not os.path.exists(label):
            #     self.image_list.pop(i)
            self.label_list.append(label)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_list[idx])
        image = prep_image(image, self.input_dim, self.ctx)
        label = prep_label(self.label_list[idx], num_classes=len(self.classes), ctx=self.ctx)
        return image.squeeze(), label.squeeze()


class LossRecorder(mx.metric.EvalMetric):
    """LossRecorder is used to record raw loss so we can observe loss directly
    """

    def __init__(self, name):
        super(LossRecorder, self).__init__(name)

    def update(self, labels, preds=0):
        """Update metric with pure loss
        """
        for loss in labels:
            if isinstance(loss, mx.nd.NDArray):
                loss = loss.asnumpy()
            self.sum_metric += loss.sum()
            self.num_inst += 1


class MyThread(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result  # 如果子线程不使用join方法，此处可能会报没有self.result的错误
        except Exception as e:
            print(e)
            return None


def calculate_label(xywh, pred_score, labels):
    label_len = labels.shape[2]
    t_y = np.zeros(shape=(xywh.shape[0], xywh.shape[1], label_len), dtype="float32")
    iou_score_single_time = 0
    for x_box in range(0, xywh.shape[0]):
        iou_score_start = time.time()
        stride = 26
        tmp_xy = (labels[x_box, :, :2] / 416.0 * stride).astype("int")
        for y_box in range(507, 2535):
            # if y_box < label_len * 3 * 13 * 13:
            #     stride = 13
            #     xy = [y_box // 3 // stride, y_box // 3 % stride]
            # elif y_box < label_len * 3 * (13 * 13 + 26 * 26):
            #     stride = 26
            #     xy = [(y_box - 507) // 3 // stride, (y_box - 507) // 3 % stride]
            # else:
            #     stride = 52
            #     xy = [(y_box - 2535) // 3 // stride, (y_box - 2535) // 3 % stride]

            xy = [(y_box - 507) // 3 // 26, (y_box - 507) // 3 % 26]
            item_index = []
            for i in range(label_len):
                if tmp_xy[i][0] == xy[0] and tmp_xy[i][1] == xy[1] and labels[x_box, i, 4] == 1.0:
                    item_index.append(i)
            if item_index:
                tmp_xywh = np.repeat(np.expand_dims(xywh[x_box, y_box, :4], axis=0), repeats=len(item_index), axis=0)
                tmp_tbox = labels[x_box, item_index, :4]
                iou_list = bbox_iou(tmp_xywh, tmp_tbox, mode="xywh")
                max_iou_score = np.max(iou_list)

                base = (y_box - 507) // 3 * 3
                bias = y_box - 507 - base

                max_iou_idx = np.argmax(iou_list, axis=0).astype("int")

                label = labels[x_box, item_index[max_iou_idx], :].copy()
                label[2:4] = np.sqrt(label[2:4])
                if max_iou_score > 0.6:
                    # pred_score[x_box, [base // 4 + bias, y_box, 2535 + base * 4 + bias]] = 1.0
                    pred_score[x_box, y_box] = 1.0
                # else:
                #     label[4] = max_iou_score
                t_y[x_box, [base // 4 + bias, y_box, 2535 + base * 4 + bias]] = label
                # t_y[x_box, y_box] = label
                # if t_y[x_box, base // 4 + bias, 4] < max_iou_score:
                #     t_y[x_box, base // 4 + bias] = label
                # if t_y[x_box, y_box, 4] < max_iou_score:
                #     t_y[x_box, y_box] = label
                # if t_y[x_box, 2535 + base * 4 + bias, 4] < max_iou_score:
                #     t_y[x_box, 2535 + base * 4 + bias] = label
        iou_score_single_time += time.time() - iou_score_start
    print("iou score single time: {}".format(iou_score_single_time))

    return t_y


if __name__ == '__main__':
    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
               "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]
    ctx = [mx.gpu(4)]
    batch_size = 12 * len(ctx)
    dataset = YoloDataSet("./data/train.txt", classes=classes, is_shuffle=True, ctx=ctx[0])
    train_data = gluon.data.DataLoader(dataset, batch_size=batch_size)
    sce_loss = gluon.loss.SigmoidBCELoss(from_sigmoid=True)
    l1_loss = gluon.loss.L1Loss()
    l2_loss = gluon.loss.L2Loss()

    obj_loss = LossRecorder('objectness_loss')
    cls_loss = LossRecorder('classification_loss')
    box_loss = LossRecorder('box_refine_loss')
    positive_weight = 1.0
    negative_weight = 0.1
    class_weight = 5.0
    box_weight = 0.01

    net = DarkNet(num_classes=len(classes))
    net.initialize(ctx=ctx)
    X = nd.uniform(shape=(2, 3, 416, 416), ctx=ctx[0])
    net(X)
    net.load_weights("./yolov3.weights")
    anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                        (59, 119), (116, 90), (156, 198), (373, 326)])
    adam = mx.optimizer.Optimizer.create_optimizer("adam")
    finetune_lr = dict({"conv_{}_weight".format(k): 0 for k in range(50)})
    adam.set_learning_rate(0.001)
    # adam.set_lr_mult(finetune_lr)
    trainer = gluon.Trainer(net.collect_params(), optimizer=adam)
    num_classes = len(classes)

    for epoch in range(200):  # reset data iterators and metrics
        cls_loss.reset()
        obj_loss.reset()
        box_loss.reset()
        tic = time.time()
        for i, batch in enumerate(train_data):
            gpu_x = split_and_load(batch[0], ctx)
            gpu_y = split_and_load(batch[1], ctx)
            record_pause = 0


            def record(t_x):
                ele_num = 3 * (5 + num_classes)
                prediction = [t_x[:, :ele_num * 13 * 13],
                              t_x[:, ele_num * 13 * 13: ele_num * (13 * 13 + 26 * 26)],
                              t_x[:, ele_num * (13 * 13 + 26 * 26):]]
                anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
                xywh = None
                score = None
                cls_pred = None
                for j in range(0, 3):
                    tmp_anchors = anchors[anchors_mask[j]]
                    tmp_xywh, tmp_score, tmp_cls = predict_transform(prediction[j], 416, tmp_anchors,
                                                                     num_classes, stride=416 // (pow(2, j) * 13))
                    if xywh is None:
                        xywh = tmp_xywh
                    else:
                        xywh = nd.concat(xywh, tmp_xywh, dim=1)
                    if score is None:
                        score = tmp_score
                    else:
                        score = nd.concat(score, tmp_score, dim=1)
                    if cls_pred is None:
                        cls_pred = tmp_cls
                    else:
                        cls_pred = nd.concat(cls_pred, tmp_cls, dim=1)
                return xywh, score, cls_pred


            with autograd.record():
                prediction_list = [record(net(t_x)) for t_x in gpu_x]
                # iou_list = []
                # for p_i in range(len(prediction_list)):
                #     thread = MyThread(calculate_label,
                #                       args=(prediction_list[p_i][0], prediction_list[p_i][1], gpu_y[p_i]))
                #     thread.start()
                #     iou_list.append(thread)
                # for thread in iou_list:
                #     thread.join()
                for p_i in range(len(prediction_list)):
                    xywh, score, cls_pred = prediction_list[p_i]
                    # tbox, tscore, tid, coordinate_weight, score_weight = iou_list[iou_i].get_result()
                    with autograd.pause():
                        t_y = calculate_label(xywh.asnumpy(), score.asnumpy(), gpu_y[p_i].asnumpy())
                        t_y = nd.array(t_y, ctx=xywh.context)
                        tbox = t_y[:, :, :4]
                        tscore = t_y[:, :, 4].reshape(0, -1, 1)
                        tid = t_y[:, :, 5:]
                        coordinate_weight = (tscore != 0.0).astype("float32")
                        score_weight = nd.where(coordinate_weight == 1.0,
                                                nd.ones_like(coordinate_weight) * positive_weight,
                                                nd.ones_like(coordinate_weight) * negative_weight)
                        tbox[:, :, 2:] = nd.sqrt(tbox[:, :, 2:])
                    xy = xywh.slice_axis(begin=0, end=2, axis=-1)
                    wh = nd.sqrt(nd.abs(xywh.slice_axis(begin=2, end=4, axis=-1)) + 0.01)
                    # total_num = 10647
                    # nonzero_count = np.count_nonzero(tscore.asnumpy(), axis=1)
                    # nonzero_count = np.where(nonzero_count > 0, nonzero_count, np.ones_like(nonzero_count))
                    # nonzero_scale = nd.array(total_num / nonzero_count, ctx=score.context) \
                    #     .reshape((-1))
                    loss1 = sce_loss(cls_pred, tid, coordinate_weight * class_weight)
                    loss2 = sce_loss(score, tscore, score_weight)
                    loss3 = l2_loss(xy, tbox.slice_axis(begin=0, end=2, axis=-1), coordinate_weight * box_weight)
                    loss4 = l2_loss(wh, tbox.slice_axis(begin=2, end=4, axis=-1), coordinate_weight * box_weight)
                    item_index = np.argwhere(tscore.asnumpy() != 0.0)
                    # print(xy[item_index[0][0], item_index[0][1]], tbox[item_index[0][0], item_index[0][1]])
                    tmp_cls = cls_pred[item_index[0][0], item_index[0][1]]
                    tmp_score = score[item_index[0][0], item_index[0][1]]
                    item_index = np.argwhere(tid.asnumpy()[item_index[0][0], item_index[0][1]] == 1.0)
                    print(tmp_cls[item_index[0]])
                    print(tmp_score)
                    # for x_box in range(cls_pred.shape[0]):
                    #     for y_box in range(cls_pred.shape[1]):
                    #         if tscore[x_box, y_box].asscalar() == 1:
                    #             print(cls_pred[x_box][y_box], tid[x_box][y_box])
                    #             flag = True
                    #             break
                    #     if flag:
                    #         break

                    loss = loss1 + loss2 + loss3 + loss4
                    cls_loss.update(loss1)
                    obj_loss.update(loss2)
                    box_loss.update(loss3)
                    loss.backward()
                loss = cls_loss.get()[1] + obj_loss.get()[1] + box_loss.get()[1]
                # net.save_params("./models/yolov3_{}_loss_{:.3f}.params".format(epoch, loss))
            trainer.step(batch_size)
            print("batch: {} / {}".format(i, np.ceil(len(dataset) / batch_size)))
            print("cls_loss: {:.5f}\nobj_loss: {:.5f}\nbox_loss: {:.5f}\n"
                  .format(cls_loss.get()[1], obj_loss.get()[1], box_loss.get()[1]))
        nd.waitall()
        print('Epoch %2d, train %s %.5f, %s %.5f, %s %.5f time %.1f sec' % (
            epoch, *cls_loss.get(), *obj_loss.get(), *box_loss.get(), time.time() - tic))
        loss = cls_loss.get()[1] + obj_loss.get()[1] + box_loss.get()[1]
        net.save_params("./models/yolov3_{}_loss_{:.3f}.params".format(epoch, loss))
