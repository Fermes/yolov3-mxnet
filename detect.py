import os
import time
from mxnet.gluon import nn
from util_mxnet import *
from darknet_mxnet import DarkNet
import mxnet as mx

def parse_cfg(cfgfile):
    """
    Takes a configuration file

    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list

    """

    file = open(cfgfile, 'r')
    lines = file.read().split('\n')  # store the lines in a list
    lines = [x for x in lines if len(x) > 0]  # get read of the empty lines
    lines = [x for x in lines if x[0] != '#']  # get rid of comments
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":  # This marks the start of a new block
            if len(block) != 0:  # If block is not empty, implies it is storing values of previous block.
                blocks.append(block)  # add it the blocks list
                block = {}  # re-init the block
            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks


class RouteLayer(nn.Block):
    def __init__(self):
        super(RouteLayer, self).__init__()


class ShortCutLayer(nn.Block):
    def __init__(self, prev):
        super(ShortCutLayer, self).__init__()
        self.prev = prev


class DetectionLayer(nn.Block):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


class UpsampleLayer(nn.Block):
    def __init__(self, scale, sample_type="nearest"):
        super(UpsampleLayer, self).__init__()
        self.scale = scale
        self.sample_type = sample_type

    def forward(self, x):
        return nd.UpSampling(x, scale=self.scale, sample_type=self.sample_type)


def create_modules(blocks):
    net_info = blocks[0]  # Captures the information about the input and pre-processing
    module_list = nn.Sequential()

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        # If it's a convolutional layer
        if (x["type"] == "convolutional"):
            # Get the info about the layer
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            # Add the convolutional layer
            conv = nn.Conv2D(filters, kernel_size=kernel_size, strides=(stride, stride), padding=(pad, pad),
                             use_bias=bias)
            module.add(conv)

            # Add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm()
                module.add(bn)

            # Check the activation.
            # It is either Linear or a Leaky ReLU for YOLO
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1)
                module.add(activn)

            # If it's an upsampling layer
            # We use Bilinear2dUpsampling
        elif x["type"] == "upsample":
            upsample = UpsampleLayer(scale=2, sample_type="nearest")
            module.add(upsample)

        # If it is a route layer
        elif x["type"] == "route":
            x["layers"] = x["layers"].split(',')
            module.add(RouteLayer())

        elif x["type"] == "shortcut":
            module.add(ShortCutLayer(prev=int(x["from"]) + index))

        # Yolo is the detection layer
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            # detection = DetectionLayer(anchors)
            # module.add_module("Detection_{}".format(index), detection)
            module.add(DetectionLayer(anchors))

        module_list.add(module)
        # prev_filters = filters
        # output_filters.append(filters)

    return net_info, module_list


class Darknet(nn.Block):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x):
        modules = self.blocks[1:]
        outputs = {}

        write = 0
        for i, module in enumerate(modules):
            module_type = (module["type"])

            print("{}  {}  \n{}\n".format(i, self.module_list[i], x.reshape((-1, ))[:6]))

            if module_type == "convolutional":
                x = self.module_list[i](x)

            elif module_type == "upsample":
                x = self.module_list[i](x)

            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]

                if (layers[0]) > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    # x = torch.cat((map1, map2), 1)
                    x = nd.concat(map1, map2, dim=1)

            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i - 1] + outputs[i + from_]

            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                # Get the input dimensions
                inp_dim = int(self.net_info["height"])

                # Get the number of classes
                num_classes = int(module["classes"])

                # Transform
                x = predict_transform(x, inp_dim, anchors, num_classes)
                if not write:
                    detections = x
                    write = 1
                else:
                    detections = nd.concat(detections, x, dim=1)

            outputs[i] = x

        return detections

    def load_weights(self, weightfile):
        # Open the weights file
        fp = open(weightfile, "rb")

        # The first 5 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4,5. Images seen by the network (during training)
        self.header = nd.array(np.fromfile(fp, dtype=np.int32, count=5))
        self.seen = self.header[3]

        weights = np.fromfile(fp, dtype=np.float32)
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]

            # If module_type is convolutional load weights
            # Otherwise ignore.

            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i + 1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]

                if batch_normalize:
                    bn = model[1]

                    # Get the number of weights of Batch Norm Layer
                    num_bn_beta = self.numel(bn.beta.shape)
                    # Load the weights
                    bn_beta = weights[ptr:ptr + num_bn_beta]
                    ptr += num_bn_beta

                    bn_gamma = weights[ptr: ptr + num_bn_beta]
                    ptr += num_bn_beta

                    bn_running_mean = weights[ptr: ptr + num_bn_beta]
                    ptr += num_bn_beta

                    bn_running_var = weights[ptr: ptr + num_bn_beta]
                    ptr += num_bn_beta

                    # Cast the loaded weights into dims of model weights.
                    bn_beta = bn_beta.reshape(bn.beta.shape)
                    bn_gamma = bn_gamma.reshape(bn.gamma.shape)
                    bn_running_mean = bn_running_mean.reshape(bn.running_mean.shape)
                    bn_running_var = bn_running_var.reshape(bn.running_var.shape)

                    bn.gamma.set_data(bn_gamma)
                    bn.beta.set_data(bn_beta)
                    bn.running_mean.set_data(bn_running_mean)
                    bn.running_var.set_data(bn_running_var)
                    # Copy the data to model
                #                     bn.bias.data.copy_(bn_biases)
                #                     bn.weight.data.copy_(bn_weights)
                #                     bn.running_mean.copy_(bn_running_mean)
                #                     bn.running_var.copy_(bn_running_var)

                else:
                    # Number of biases
                    num_biases = self.numel(conv.bias.shape)

                    # Load the weights
                    conv_biases = weights[ptr: ptr + num_biases]
                    ptr = ptr + num_biases

                    # reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.reshape(conv.bias.shape)

                    # Finally copy the data
                    # conv.bias.data.copy_(conv_biases)
                    conv.bias.set_data(conv_biases)

                # Let us load the weights for the Convolutional layers
                num_weights = self.numel(conv.weight.shape)

                # Do the same as above for weights
                conv_weights = weights[ptr:ptr + num_weights]
                ptr = ptr + num_weights

                conv_weights = conv_weights.reshape(conv.weight.shape)
                # conv.weight.data.copy_(conv_weights)

                conv.weight.set_data(conv_weights)

    def numel(self, x):
        if isinstance(x, nd.NDArray):
            x.asnumpy()
        return np.prod(x)


def draw_bbox(img, bboxs):
    for x in bboxs:
        c1 = tuple(x[1:3].astype(np.int))
        c2 = tuple(x[3:5].astype(np.int))
        cls = int(x[-1])
        # color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        color = (255, 0, 0)
        label = "{0}".format(classes[cls])
        cv2.rectangle(img, c1, c2, color, 2)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c2 = c1[0] + t_size[0] + 2, c1[1] - t_size[1] - 5
        cv2.rectangle(img, c1, c2, color, -1)
        cv2.putText(img, label, (c1[0], c1[1] - t_size[1] + 7), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
    return img


def save_results(load_images, output):
    im_dim_list = nd.array([(x.shape[1], x.shape[0]) for x in load_images])
    im_dim_list = nd.tile(im_dim_list, 2)
    im_dim_list = im_dim_list[output[:, 0], :]

    scaling_factor = nd.min(416 / im_dim_list).asscalar()
    # scaling_factor = (416 / im_dim_list)[0].view(-1, 1)

    output[:, [1, 3]] -= (416 - scaling_factor * im_dim_list[:, 0].reshape((-1, 1))) / 2
    output[:, [2, 4]] -= (416 - scaling_factor * im_dim_list[:, 1].reshape((-1, 1))) / 2
    output[:, 1:5] /= scaling_factor

    for i in range(output.shape[0]):
        output[i, [1, 3]] = nd.clip(output[i, [1, 3]], a_min=0.0, a_max=im_dim_list[i, 0].asscalar())
        output[i, [2, 4]] = nd.clip(output[i, [2, 4]], a_min=0.0, a_max=im_dim_list[i, 1].asscalar())

    output_recast = time.time()
    class_load = time.time()
    draw = time.time()
    output = output.asnumpy()
    # load_images[0] = draw_bbox(load_images[0], output)
    for i in range(len(load_images)):
        bboxs = []
        for bbox in output:
            if i == int(bbox[0]):
                bboxs.append(bbox)
        draw_bbox(load_images[i], bboxs)
    # list(map(lambda x: draw_bbox(load_images, x), results))
    list(map(cv2.imwrite, [os.path.join(dst_dir, "{0}.jpg".format(i)) for i in range(len(load_images))], load_images))

    end = time.time()

    print("SUMMARY")
    print("----------------------------------------------------------")
    print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
    print()
    print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
    print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
    print("----------------------------------------------------------")


if __name__ == '__main__':
    images = "./images"
    batch_size = 1
    confidence = 0.5
    nms_thresh = 0.4
    dst_dir = "./results"
    start = 0
    classes = load_classes("data/coco.names")
    # classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
    #            "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
    #            "sheep", "sofa", "train", "tvmonitor"]
    num_classes = len(classes)
    net = DarkNet(num_classes=num_classes)
    net.initialize()
    input_dim = 416
    ctx = mx.cpu()
    try:
        imlist = [os.path.join(images, img) for img in os.listdir(images)]
    except NotADirectoryError:
        imlist = []
        imlist.append(images)
    except FileNotFoundError:
        print("No file or directory with the name {}".format(images))

    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    load_images = [cv2.imread(img) for img in imlist]

    im_batches = list(map(prep_image, load_images, [input_dim for x in range(len(imlist))]))
    im_dim_list = nd.array([(x.shape[1], x.shape[0]) for x in load_images])
    im_dim_list = nd.tile(im_dim_list, 2)

    output = net(im_batches[0].expand_dims(0))
    # for module in net.collect_params():
    #   module.initialize(force_reinit=True)
    net.load_weights("yolov3.weights")
    # net.hybridize()
    # net.load_params("models/yolov3_4_loss_46.212.params")

    write = 0

    start_det_loop = time.time()
    anchors = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                        (59, 119), (116, 90), (156, 198), (373, 326)]
    for i, batch in enumerate(im_batches):
        # load the image
        start = time.time()

        prediction = net(batch.expand_dims(0))
        ele_num = 3 * (5 + num_classes)
        prediction = [prediction[:, :ele_num * 13 * 13],
                      prediction[:, ele_num * 13 * 13: ele_num * (13 * 13 + 26 * 26)],
                      prediction[:, ele_num * (13 * 13 + 26 * 26):]]
        for j in range(3):
            base = (3 - j) * 3
            tmp_anchors = [anchors[base - 3], anchors[base - 2], anchors[base - 1]]
            xywh, score, cls = predict_transform(prediction[j], 416, tmp_anchors,
                                                 num_classes, stride=416//(pow(2, j) * 13), is_train=False)
            prediction[j] = nd.concat(xywh, score, cls, dim=2)
            # prediction[j] = predict_transform(prediction[j], 416, tmp_anchors,
            #                                   num_classes, stride=416//(pow(2, j) * 13))
        prediction = nd.concat(prediction[0], prediction[1], prediction[2], dim=1)
        prediction = write_results(prediction, confidence, num_classes, nms_conf=nms_thresh)

        end = time.time()

        if type(prediction) == int:

            for im_num, image in enumerate(imlist[i * batch_size: min((i + 1) * batch_size, len(imlist))]):
                im_id = i * batch_size + im_num
                print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start) / batch_size))
                print("----------------------------------------------------------")
            continue

        prediction[:, 0] += i * batch_size  # transform the atribute from index in batch to index in imlist

        if not write:  # If we have't initialised output
            output = prediction
            write = 1
        else:
            output = nd.concat(output, prediction, dim=0)

        for im_num, image in enumerate(imlist[i * batch_size: min((i + 1) * batch_size, len(imlist))]):
            im_id = i * batch_size + im_num
            objs = [classes[int(x.asnumpy()[-1])] for x in output if int(x.asnumpy()[0]) == im_id]
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start) / batch_size))
            print("----------------------------------------------------------")

    try:
        output
        save_results(load_images, output)
    except NameError:
        print("No detections were made")
        exit()
    except Exception as e:
        print(e)
