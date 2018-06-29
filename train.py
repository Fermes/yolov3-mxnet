import re
import threading
import time
import os
import xml.etree.ElementTree as ET
from random import shuffle
from mxnet import autograd
from mxnet import gluon
from darknet_mxnet import DarkNet
from util_mxnet import *
import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description="YOLO v3 Detection Module")
    parser.add_argument("--image_path", dest='image_path', help=
    "Image / Directory containing images to perform detection upon",
                        default="data/train.txt", type=str)
    parser.add_argument("--lr", dest="lr", help="learning rate", default=1e-3, type=float)
    parser.add_argument("--classes", dest="classes", default="data/coco.names", type=str)
    parser.add_argument("--prefix", dest="prefix", default="voc")
    parser.add_argument("--gpu", dest="gpu", help="gpu id", default=0, type=int)
    parser.add_argument("--dst_dir", dest='dst_dir', help=
    "Image / Directory to store detections to",
                        default="results", type=str)
    parser.add_argument("--epoch", dest="epoch", default=200, type=int)
    parser.add_argument("--batch_size", dest="batch_size", help="Batch size", default=16, type=int)
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.7, type=float)
    parser.add_argument("--params", dest='params', help=
    "mxnet params file", type=str)
    parser.add_argument("--input_dim", dest='input_dim', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default=416, type=int)

    return parser.parse_args()


def parse_xml(xml_file, classes):
    root = ET.parse(xml_file).getroot()
    image_size = root.find("size")
    size = {
        "width": float(image_size.find("width").text),
        "height": float(image_size.find("height").text),
        "depth": float(image_size.find("depth").text)
    }
    bbox = []
    if not isinstance(classes, np.ndarray):
        classes = np.array(classes)
    for obj in root.findall("object"):
        cls = np.argwhere(classes == obj.find("name").text).reshape(-1)[0]
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
        center_x = (xmin + xmax) / 2.0 / size["width"]
        center_y = (ymin + ymax) / 2.0 / size["height"]
        width = (xmax - xmin) / size["width"]
        height = (ymax - ymin) / size["height"]
        bbox.append([cls, center_x, center_y, width, height])
    return np.array(bbox)


def prep_label(label_file, classes, ctx):
    num_classes = len(classes)
    if label_file.endswith(".txt"):
        with open(label_file, "r") as file:
            labels = file.readlines()
            labels = np.array([list(map(float, x.split())) for x in labels], dtype="float32")
    elif label_file.endswith(".xml"):
        labels = parse_xml(label_file, classes)
    final_labels = nd.zeros(shape=(30, num_classes + 5), dtype="float32", ctx=ctx)
    i = 0
    for label in labels:
        one_hot = np.zeros(shape=(num_classes + 5), dtype="float32")
        one_hot[5 + int(label[0])] = 1.0
        one_hot[4] = 1.0
        one_hot[:4] = label[1:]
        final_labels[i] = one_hot
        i += 1
        i %= 30
    return nd.array(final_labels, ctx=ctx)


def prep_final_label(labels, num_classes, input_dim, ctx):
    if isinstance(labels, nd.NDArray):
        labels = labels.asnumpy()
    anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                        (59, 119), (116, 90), (156, 198), (373, 326)])
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    batch_size = labels.shape[0]
    label_1 = np.zeros(shape=(batch_size, 13, 13, 3, num_classes + 5), dtype="float32")
    label_2 = np.zeros(shape=(batch_size, 26, 26, 3, num_classes + 5), dtype="float32")
    label_3 = np.zeros(shape=(batch_size, 52, 52, 3, num_classes + 5), dtype="float32")

    true_label_1 = np.zeros(shape=(batch_size, 13, 13, 3, 5), dtype="float32")
    true_label_2 = np.zeros(shape=(batch_size, 26, 26, 3, 5), dtype="float32")
    true_label_3 = np.zeros(shape=(batch_size, 52, 52, 3, 5), dtype="float32")
    label_list = [label_1, label_2, label_3]
    true_label_list = [true_label_1, true_label_2, true_label_3]
    for x_box in range(labels.shape[0]):
        for y_box in range(labels.shape[1]):
            if labels[x_box, y_box, 4] == 0.0:
                break
            for i in range(3):
                stride = np.power(2, i) * 13
                tmp_anchors = anchors[anchors_mask[i]]
                tmp_xywh = np.repeat(np.expand_dims(labels[x_box, y_box, :4] * stride, axis=0),
                                     repeats=tmp_anchors.shape[0], axis=0)
                anchor_xywh = tmp_xywh.copy()
                anchor_xywh[:, 2:4] = tmp_anchors / input_dim * stride
                best_anchor = np.argmax(bbox_iou(tmp_xywh, anchor_xywh), axis=0)
                label = labels[x_box, y_box].copy()
                tmp_idx = (label[:2] * stride).astype("int")
                label[:2] = label[:2] * stride
                label[:2] -= tmp_idx
                label[2:4] = np.log(label[2:4] * input_dim / tmp_anchors[best_anchor] + 1e-16)

                label_list[i][x_box, tmp_idx[1], tmp_idx[0], best_anchor] = label

                true_xywhs = labels[x_box, y_box, :5] * input_dim
                true_xywhs[4] = 1.0
                true_label_list[i][x_box, tmp_idx[1], tmp_idx[0], best_anchor] = true_xywhs
    t_y = nd.concat(nd.array(label_1.reshape((batch_size, -1, num_classes + 5)), ctx=ctx),
                    nd.array(label_2.reshape((batch_size, -1, num_classes + 5)), ctx=ctx),
                    nd.array(label_3.reshape((batch_size, -1, num_classes + 5)), ctx=ctx),
                    dim=1)
    t_xywhs = nd.concat(nd.array(true_label_1.reshape((batch_size, -1, 5)), ctx=ctx),
                        nd.array(true_label_2.reshape((batch_size, -1, 5)), ctx=ctx),
                        nd.array(true_label_3.reshape((batch_size, -1, 5)), ctx=ctx),
                        dim=1)

    return t_y, t_xywhs


def calculate_ignore(prediction, true_xywhs):
    if isinstance(true_xywhs, nd.NDArray):
        true_xywhs = true_xywhs.asnumpy()
    prediction = predict_transform(prediction, input_dim, anchors).asnumpy()
    ignore_mask = np.ones(shape=pred_score.shape, dtype="float32")
    item_index = np.argwhere(true_xywhs[:, :, 4] == 1.0)

    for x_box, y_box in item_index:
        iou = bbox_iou(prediction[x_box, y_box:y_box + 1, :4], true_xywhs[x_box, y_box:y_box + 1]) < 0.7
        ignore_mask[x_box, y_box:y_box + 1] = iou.astype("float32").reshape((-1, 1))
    return ignore_mask


class YoloDataSet(gluon.data.Dataset):
    def __init__(self, images_path, classes, input_dim=416, is_shuffle=False, ctx=mx.cpu()):
        super(YoloDataSet, self).__init__()
        self.anchors = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                        (59, 119), (116, 90), (156, 198), (373, 326)]
        self.classes = classes
        self.input_dim = input_dim
        self.ctx = ctx
        self.label_mode = "xml"
        if os.path.isdir(images_path):
            images_path = os.path.join(images_path, "train")
            self.image_list = os.listdir(images_path)
            self.image_list = [os.path.join(images_path, im.strip()) for im in self.image_list]
        elif images_path.endswith(".txt"):
            self.label_mode = "txt"
            with open(images_path, "r") as file:
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
            if self.label_mode == "txt":
                label = pattern.sub(lambda s: ".txt", self.image_list[i]).replace("JPEGImages", "labels")
            else:
                label = pattern.sub(lambda s: ".xml", self.image_list[i]).replace("train", "train_label")
            if not os.path.exists(label):
                self.image_list.pop(i)
                continue
            self.label_list.append(label)
        self.label_list.reverse()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_list[idx])
        image = nd.array(prep_image(image, self.input_dim), ctx=self.ctx)
        label = prep_label(self.label_list[idx], classes=self.classes, ctx=self.ctx)
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


class SigmoidBinaryCrossEntropyLoss(gluon.loss.Loss):
    def __init__(self, from_sigmoid=False, weight=None, batch_axis=0, **kwargs):
        super(SigmoidBinaryCrossEntropyLoss, self).__init__(weight, batch_axis, **kwargs)
        self._from_sigmoid = from_sigmoid

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        label = gluon.loss._reshape_like(F, label, pred)
        if not self._from_sigmoid:
            # We use the stable formula: max(x, 0) - x * z + log(1 + exp(-abs(x)))
            tmp_loss = F.relu(pred) - pred * label + F.Activation(-F.abs(pred), act_type='softrelu')
        else:
            tmp_loss = -(F.log(pred+1e-12)*label + F.log(1.-pred+1e-12)*(1.-label))
        tmp_loss = gluon.loss._apply_weighting(F, tmp_loss, self._weight, sample_weight)
        return tmp_loss


class L1Loss(gluon.loss.Loss):
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(L1Loss, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        label = gluon.loss._reshape_like(F, label, pred)
        tmp_loss = F.abs(pred - label)
        tmp_loss = gluon.loss._apply_weighting(F, tmp_loss, self._weight, sample_weight)
        return tmp_loss


class L2Loss(gluon.loss.Loss):
    def __init__(self, weight=1., batch_axis=0, **kwargs):
        super(L2Loss, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        label = gluon.loss._reshape_like(F, label, pred)
        tmp_loss = F.square(pred - label)
        tmp_loss = gluon.loss._apply_weighting(F, tmp_loss, self._weight / 2, sample_weight)
        return tmp_loss


if __name__ == '__main__':
    args = arg_parse()
    classes = load_classes(args.classes)
    num_classes = len(classes)
    ctx = try_gpu(args.gpu)
    input_dim = args.input_dim
    batch_size = args.batch_size
    dataset = YoloDataSet(args.image_path, classes=classes, is_shuffle=True, ctx=ctx)
    train_data = gluon.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    obj_loss = LossRecorder('objectness_loss')
    cls_loss = LossRecorder('classification_loss')
    box_loss = LossRecorder('box_refine_loss')
    positive_weight = 1.0
    negative_weight = 1.0

    sce_loss = SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
    l1_loss = L1Loss()
    l2_loss = L2Loss()

    net = DarkNet(num_classes=num_classes, input_dim=input_dim)
    net.initialize(ctx=ctx)
    if args.params:
        net.load_params(args.params)
        print("load params: {}".format(args.params))
    else:
        X = nd.uniform(shape=(1, 3, input_dim, input_dim), ctx=ctx)
        net(X)
        net.load_weights("./data/yolov3.weights", fine_tune=True)
    # for _, w in net.collect_params().items():
    #     if w.name.find("58") == -1 and w.name.find("66") == -1 and w.name.find("74") == -1:
    #         w.grad_req = "null"
    anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                        (59, 119), (116, 90), (156, 198), (373, 326)])
    adam = mx.optimizer.Optimizer.create_optimizer("adam")
    finetune_lr = dict({"conv_{}_weight".format(k): 1 for k in [58, 66, 74]})
    lr2 = dict({"conv_{}_bias".format(k): 2 for k in [58, 66, 74]})
    adam.set_learning_rate(args.lr)
    adam.set_lr_mult({**finetune_lr, **lr2})
    trainer = gluon.Trainer(net.collect_params(), optimizer=adam)

    cur_loss = 10000000.
    early_stop = 0
    for epoch in range(args.epoch):  # reset data iterators and metrics
        cls_loss.reset()
        obj_loss.reset()
        box_loss.reset()
        tic = time.time()
        for i, batch in enumerate(train_data):
            gpu_x = batch[0]
            gpu_y = batch[1]
            record_pause = 0

            with autograd.record():
                prediction = net(gpu_x)
                pred_xy = prediction[:, :, :2]
                pred_wh = prediction[:, :, 2:4]
                pred_score = prediction[:, :, 4:5]
                pred_cls = prediction[:, :, 5:]
                with autograd.pause():
                    t_y, true_xywhs = prep_final_label(gpu_y, num_classes, input_dim, ctx=pred_xy.context)
                    ignore_mask = nd.array(calculate_ignore(prediction.copy(), true_xywhs)
                                           , ctx=pred_xy.context)
                    true_box = t_y[:, :, :4]
                    true_score = t_y[:, :, 4:5]
                    true_cls = t_y[:, :, 5:]
                    coordinate_weight = true_score.copy()
                    score_weight = nd.where(coordinate_weight == 1.0,
                                            nd.ones_like(coordinate_weight) * positive_weight,
                                            nd.ones_like(coordinate_weight) * negative_weight)

                box_loss_scale = 2. - true_xywhs[:, :, 2:3] * true_xywhs[:, :, 3:4] / float(args.input_dim ** 2)

                loss_xy = sce_loss(pred_xy, true_box.slice_axis(begin=0, end=2, axis=-1),
                                  ignore_mask * coordinate_weight * box_loss_scale)
                loss_wh = l1_loss(pred_wh, true_box.slice_axis(begin=2, end=4, axis=-1),
                                  ignore_mask * coordinate_weight * 0.5 * box_loss_scale)
                loss_conf = sce_loss(pred_score, true_score, score_weight)

                loss_cls = sce_loss(pred_cls, true_cls, coordinate_weight)

                value_num = nd.sum(coordinate_weight, axis=(1, 2))
                total_num = nd.sum(coordinate_weight)

                t_loss_xy = nd.sum(loss_xy, axis=(1, 2)) / value_num / 2
                t_loss_wh = nd.sum(loss_wh, axis=(1, 2)) / value_num / 2

                one_loss_conf = loss_conf * coordinate_weight
                zero_loss_conf = loss_conf * (1. - coordinate_weight)
                # zero_loss_conf = nd.where(loss_conf * (1. - coordinate_weight) < 0.05,
                #                           nd.zeros_like(loss_conf),
                #                           loss_conf)
                zero_num_conf = nd.sum(zero_loss_conf != 0., axis=(1, 2))
                t_loss_conf = nd.sum(one_loss_conf, axis=(1, 2)) / value_num + \
                              nd.sum(zero_loss_conf, axis=(1, 2)) / zero_num_conf
                # t_loss_cls = nd.sum(loss_cls, axis=(1, 2)) / num_classes / value_num
                one_loss_cls = loss_cls * true_cls
                one_num_cls = nd.sum(true_cls, axis=(1, 2))
                zero_loss_cls = loss_cls * (1. - true_cls) * coordinate_weight
                zero_num_cls = nd.sum((1. - true_cls) * coordinate_weight, axis=(1, 2))
                t_loss_cls = nd.sum(one_loss_cls, axis=(1, 2)) / one_num_cls + nd.sum(zero_loss_cls, axis=(1, 2)) / zero_num_cls

                # loss = nd.concat(loss_xy, loss_wh, loss_conf, loss_cls, dim=-1)
                loss = t_loss_xy + t_loss_wh + t_loss_conf + t_loss_cls
                loss.backward()

                cls_loss.update(t_loss_cls)
                obj_loss.update(t_loss_conf)
                box_loss.update(t_loss_xy + t_loss_wh)
                item_index = np.nonzero(true_score.asnumpy())
                print((nd.sum(pred_score > 0.5) / total_num).asscalar())
                print((nd.sum(nd.abs(pred_score * coordinate_weight - true_score)) / total_num).asscalar())
                print(nd.sum(nd.abs(pred_wh * coordinate_weight - t_y[:, :, 2:4]))
                      / 2 / total_num)
                print("cls_loss: {:.5f}\nobj_loss: {:.5f}\nbox_loss: {:.5f}\n"
                      .format(nd.mean(t_loss_cls).asscalar(), nd.mean(t_loss_conf).asscalar(),
                              nd.mean(t_loss_xy + t_loss_wh).asscalar()))
            trainer.step(batch_size)
            print("batch: {} / {}".format(i, np.ceil(len(dataset) / batch_size)))
        nd.waitall()
        print('Epoch %2d, train %s %.5f, %s %.5f, %s %.5f time %.1f sec' % (
            epoch, *cls_loss.get(), *obj_loss.get(), *box_loss.get(), time.time() - tic))
        loss = cls_loss.get()[1] + obj_loss.get()[1] + box_loss.get()[1]
        if loss < cur_loss:
            early_stop = 0
            cur_loss = loss
            net.save_params("./models/{0}_{1}_loss_{2:.3f}.params".format(args.prefix, epoch, cur_loss))
        else:
            early_stop += 1
            if early_stop >= 10:
                print("train stop, epoch: {0}  loss: {1:.3f}".format(epoch, cur_loss))
                exit()
