import re
import threading
import time
import xml.etree.ElementTree as ET
from random import shuffle
from mxnet import autograd
from mxnet import gluon
from darknet_mxnet import DarkNet
from util_mxnet import *


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
        final_labels = nd.zeros(shape=(30, num_classes +5), dtype="float32", ctx=ctx)
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
        label = prep_label(self.label_list[idx], len(self.classes), self.ctx)
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


def calculate_label(t_xywh, t_labels):
    xywh = t_xywh.asnumpy()
    labels = t_labels.asnumpy()
    label_len = labels.shape[2]
    empty_label = np.zeros(shape=label_len)
    t_y = np.zeros(shape=(xywh.shape[0], xywh.shape[1], label_len), dtype="float32")
    iou_score_single_time = 0
    empty_num = 0
    for x_box in range(0, xywh.shape[0]):
        iou_score_start = time.time()
        for y_box in range(0, xywh.shape[1]):
            tmp_xywh = np.repeat(np.expand_dims(xywh[x_box, y_box, :4], axis=0), repeats=30, axis=0)
            tmp_tbox = labels[x_box, :, :4]
            iou_list = bbox_iou(tmp_xywh, tmp_tbox, mode="xywh")

            if y_box < label_len * 3 * 13 * 13:
                stride = 13
            elif y_box < label_len * 3 * (13 * 13 + 26 * 26):
                stride = 26
            else:
                stride = 52
            max_iou_score = np.max(iou_list)

            if max_iou_score == 0:
                t_y[x_box, y_box] = empty_label.copy()
                empty_num += 1
            else:
                max_iou_idx = np.argmax(iou_list, axis=0).astype("int")
                x, y = (xywh[x_box, y_box, :2] / 416.0 * stride).astype("int")
                tmp_x, tmp_y = (labels[x_box, max_iou_idx][:2] / 416.0 * stride).astype("int")
                if x == tmp_x and y == tmp_y:
                    t_y[x_box, y_box] = labels[x_box, max_iou_idx].copy()
                else:
                    t_y[x_box, y_box] = empty_label.copy()
        iou_score_single_time += time.time() - iou_score_start
    print("iou score single time: {}".format(iou_score_single_time))
    t_y = nd.array(t_y, ctx=t_xywh.context)
    tbox = t_y[:, :, :4]
    tscore = t_y[:, :, 4].reshape(0, -1, 1)
    tid = t_y[:, :, 5:]
    coordinate_weight = nd.ones(shape=tscore.shape, dtype="float32", ctx=t_y.context) \
                        * (tscore[:, :, 0] != 0).reshape(0, -1, 1)
    score_weight = nd.where(coordinate_weight > 0,
                            nd.ones_like(coordinate_weight) * positive_weight,
                            nd.ones_like(coordinate_weight) * negative_weight)
    tbox[:, :, 2:] = nd.sqrt(tbox[:, :, 2:])
    return tbox, tscore, tid, coordinate_weight, score_weight


if __name__ == '__main__':
    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
               "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]
    ctx = [mx.gpu(4)]
    batch_size = 32 * len(ctx)
    dataset = YoloDataSet("./data/train.txt", classes=classes, is_shuffle=True, ctx=ctx[0])
    train_data = gluon.data.DataLoader(dataset, batch_size=batch_size)
    sce_loss = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
    l1_loss = gluon.loss.L1Loss()
    l2_loss = gluon.loss.L2Loss()

    obj_loss = LossRecorder('objectness_loss')
    cls_loss = LossRecorder('classification_loss')
    box_loss = LossRecorder('box_refine_loss')
    positive_weight = 5.0
    negative_weight = 0.5
    class_weight = 1.0
    box_weight = 5.0

    net = DarkNet(num_classes=len(classes))
    net.initialize(ctx=ctx)
    X = nd.uniform(shape=(2, 3, 416, 416), ctx=ctx[0])
    net(X)
    net.load_weights("./yolov3.weights")

    anchors = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
               (59, 119), (116, 90), (156, 198), (373, 326)]
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.001, 'wd': 5e-4})
    num_classes = len(classes)

    for epoch in range(20):
        # reset data iterators and metrics
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
                tmp_anchors = [anchors[6], anchors[7], anchors[8]]
                xywh, score, cls_pred = predict_transform(prediction[0], 416, tmp_anchors,
                                                          num_classes, stride=32)
                for j in range(1, 3):
                    base = (3 - j) * 3
                    tmp_anchors = [anchors[base - 3], anchors[base - 2], anchors[base - 1]]
                    tmp_xywh, tmp_score, tmp_cls = predict_transform(prediction[j], 416, tmp_anchors,
                                                                     num_classes, stride=416 // (pow(2, j) * 13))
                    xywh = nd.concat(xywh, tmp_xywh, dim=1)
                    score = nd.concat(score, tmp_score, dim=1)
                    cls_pred = nd.concat(cls_pred, tmp_cls, dim=1)
                return xywh, score, cls_pred


            with autograd.record():
                prediction_list = [record(net(t_x)) for t_x in gpu_x]
                iou_list = []
                with autograd.pause():
                    for p_i in range(len(prediction_list)):
                        thread = MyThread(calculate_label, args=(prediction_list[p_i][0], gpu_y[p_i]))
                        thread.start()
                        iou_list.append(thread)
                    for thread in iou_list:
                        thread.join()
                for iou_i in range(len(iou_list)):
                    xywh, score, cls_pred = prediction_list[iou_i]
                    tbox, tscore, tid, coordinate_weight, score_weight = iou_list[iou_i].get_result()

                    # loss1 = sce_loss(cls_pred, tid, coordinate_weight * class_weight)
                    loss1 = sce_loss(cls_pred, tid)
                    loss2 = l1_loss(score, tscore, score_weight)
                    loss3 = l2_loss(xywh, tbox, coordinate_weight * box_weight)
                    loss = loss1 + loss2 + loss3
                    cls_loss.update(loss1)
                    obj_loss.update(loss2)
                    box_loss.update(loss3)
                    loss.backward()
            trainer.step(batch_size)
            print("batch: {} / {}".format(i, np.ceil(len(dataset) / batch_size)))
            print("cls_loss: {:.5f} \n obj_loss: {:.5f} \n box_loss: {:.5f}"
                  .format(cls_loss.get()[1], obj_loss.get()[1], box_loss.get()[1]))
        nd.waitall()
        print('Epoch %2d, train %s %.5f, %s %.5f, %s %.5f time %.1f sec' % (
            epoch, *cls_loss.get(), *obj_loss.get(), *box_loss.get(), time.time() - tic))
        loss = cls_loss.get()[1] + obj_loss.get()[1] + box_loss.get()[1]
        net.save_params("./models/yolov3_{}_loss_{:.3f}.params".format(epoch, loss))
