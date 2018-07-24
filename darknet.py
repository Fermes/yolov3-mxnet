from mxnet.gluon import nn
from utils import *


def ConvBNBlock(channels, kernel_size, strides, pad, use_bias=False, leaky=True):
    blk = nn.HybridSequential()
    blk.add(nn.Conv2D(int(channels), kernel_size=kernel_size, strides=strides, padding=pad,
                      use_bias=use_bias))
    if not use_bias:
        blk.add(nn.BatchNorm(in_channels=int(channels)))
    if leaky:
        blk.add(nn.LeakyReLU(0.1))
    return blk


class ShortCutBlock(nn.HybridBlock):
    def __init__(self, channels):
        super(ShortCutBlock, self).__init__()
        self.conv_1 = ConvBNBlock(channels / 2, 1, 1, 0)
        self.conv_2 = ConvBNBlock(channels, 3, 1, 1)

    def hybrid_forward(self, F, x, *args, **kwargs):
        blk = self.conv_1(x)
        blk = self.conv_2(blk)
        return blk + x


class UpSampleBlock(nn.HybridBlock):
    def __init__(self, scale, sample_type="nearest"):
        super(UpSampleBlock, self).__init__()
        self.scale = scale
        self.sample_type = sample_type

    def hybrid_forward(self, F, x, *args, **kwargs):
        return F.UpSampling(x, scale=self.scale, sample_type=self.sample_type)


class TransformBlock(nn.HybridBlock):
    def __init__(self, num_classes, stride):
        super(TransformBlock, self).__init__()
        self.bbox_attrs = 5 + num_classes
        self.stride = stride

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = F.transpose(x.reshape((0, self.bbox_attrs * 3, self.stride * self.stride)), (0, 2, 1)) \
            .reshape((0, self.stride * self.stride * 3, self.bbox_attrs))
        xy_pred = F.sigmoid(x.slice_axis(begin=0, end=2, axis=-1))
        wh_pred = x.slice_axis(begin=2, end=4, axis=-1)
        score_pred = F.sigmoid(x.slice_axis(begin=4, end=5, axis=-1))
        cls_pred = F.sigmoid(x.slice_axis(begin=5, end=None, axis=-1))

        return F.concat(xy_pred, wh_pred, score_pred, cls_pred, dim=-1)


class DarkNet(nn.HybridBlock):
    def __init__(self, num_classes=80, input_dim=416):
        super(DarkNet, self).__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.anchors = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                        (59, 119), (116, 90), (156, 198), (373, 326)]

        self.conv_bn_block_0 = ConvBNBlock(32, 3, 1, 1)
        self.conv_bn_block_1 = ConvBNBlock(64, 3, 2, 1)
        self.shortcut_block_4 = ShortCutBlock(64)
        self.conv_bn_block_5 = ConvBNBlock(128, 3, 2, 1)
        self.shortcut_block_8 = ShortCutBlock(128)
        self.shortcut_block_11 = ShortCutBlock(128)
        self.conv_bn_block_12 = ConvBNBlock(256, 3, 2, 1)
        self.shortcut_block_15 = ShortCutBlock(256)
        self.shortcut_block_18 = ShortCutBlock(256)
        self.shortcut_block_21 = ShortCutBlock(256)
        self.shortcut_block_24 = ShortCutBlock(256)
        self.shortcut_block_27 = ShortCutBlock(256)
        self.shortcut_block_30 = ShortCutBlock(256)
        self.shortcut_block_33 = ShortCutBlock(256)
        self.shortcut_block_36 = ShortCutBlock(256)
        self.conv_bn_block_37 = ConvBNBlock(512, 3, 2, 1)
        self.shortcut_block_40 = ShortCutBlock(512)
        self.shortcut_block_43 = ShortCutBlock(512)
        self.shortcut_block_46 = ShortCutBlock(512)
        self.shortcut_block_49 = ShortCutBlock(512)
        self.shortcut_block_52 = ShortCutBlock(512)
        self.shortcut_block_55 = ShortCutBlock(512)
        self.shortcut_block_58 = ShortCutBlock(512)
        self.shortcut_block_61 = ShortCutBlock(512)
        self.conv_bn_block_62 = ConvBNBlock(1024, 3, 2, 1)
        self.shortcut_block_65 = ShortCutBlock(1024)
        self.shortcut_block_68 = ShortCutBlock(1024)
        self.shortcut_block_71 = ShortCutBlock(1024)
        self.shortcut_block_74 = ShortCutBlock(1024)
        self.conv_bn_block_75 = ConvBNBlock(512, 1, 1, 0)
        self.conv_bn_block_76 = ConvBNBlock(1024, 3, 1, 1)
        self.conv_bn_block_77 = ConvBNBlock(512, 1, 1, 0)
        self.conv_bn_block_78 = ConvBNBlock(1024, 3, 1, 1)
        self.conv_bn_block_79 = ConvBNBlock(512, 1, 1, 0)
        self.conv_bn_block_80 = ConvBNBlock(1024, 3, 1, 1)
        self.conv_bn_block_81 = ConvBNBlock(3 * (5+self.num_classes), 1, 1, 0, use_bias=True, leaky=False)
        self.transform_0 = TransformBlock(num_classes, 13)
        self.conv_bn_block_84 = ConvBNBlock(256, 1, 1, 0)
        self.upsample_block_85 = UpSampleBlock(scale=2)
        self.conv_bn_block_87 = ConvBNBlock(256, 1, 1, 0)
        self.conv_bn_block_88 = ConvBNBlock(512, 3, 1, 1)
        self.conv_bn_block_89 = ConvBNBlock(256, 1, 1, 0)
        self.conv_bn_block_90 = ConvBNBlock(512, 3, 1, 1)
        self.conv_bn_block_91 = ConvBNBlock(256, 1, 1, 0)
        self.conv_bn_block_92 = ConvBNBlock(512, 3, 1, 1)
        self.conv_bn_block_93 = ConvBNBlock(3 * (5+self.num_classes), 1, 1, 0, use_bias=True, leaky=False)
        self.transform_1 = TransformBlock(num_classes, 26)
        self.conv_bn_block_96 = ConvBNBlock(128, 1, 1, 0)
        self.upsample_block_97 = UpSampleBlock(scale=2)
        self.conv_bn_block_99 = ConvBNBlock(128, 1, 1, 0)
        self.conv_bn_block_100 = ConvBNBlock(256, 3, 1, 1)
        self.conv_bn_block_101 = ConvBNBlock(128, 1, 1, 0)
        self.conv_bn_block_102 = ConvBNBlock(256, 3, 1, 1)
        self.conv_bn_block_103 = ConvBNBlock(128, 1, 1, 0)
        self.conv_bn_block_104 = ConvBNBlock(256, 3, 1, 1)
        self.conv_bn_block_105 = ConvBNBlock(3 * (5+self.num_classes), 1, 1, 0, use_bias=True, leaky=False)
        self.transform_2 = TransformBlock(num_classes, 52)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.conv_bn_block_0(x)
        x = self.conv_bn_block_1(x)
        x = self.shortcut_block_4(x)
        x = self.conv_bn_block_5(x)
        x = self.shortcut_block_8(x)
        x = self.shortcut_block_11(x)
        x = self.conv_bn_block_12(x)
        x = self.shortcut_block_15(x)
        x = self.shortcut_block_18(x)
        x = self.shortcut_block_21(x)
        x = self.shortcut_block_24(x)
        x = self.shortcut_block_27(x)
        x = self.shortcut_block_30(x)
        x = self.shortcut_block_33(x)
        shortcut_36 = self.shortcut_block_36(x)
        x = self.conv_bn_block_37(shortcut_36)
        x = self.shortcut_block_40(x)
        x = self.shortcut_block_43(x)
        x = self.shortcut_block_46(x)
        x = self.shortcut_block_49(x)
        x = self.shortcut_block_52(x)
        x = self.shortcut_block_55(x)
        x = self.shortcut_block_58(x)
        shortcut_61 = self.shortcut_block_61(x)
        x = self.conv_bn_block_62(shortcut_61)
        x = self.shortcut_block_65(x)
        x = self.shortcut_block_68(x)
        x = self.shortcut_block_71(x)
        x = self.shortcut_block_74(x)
        x = self.conv_bn_block_75(x)
        x = self.conv_bn_block_76(x)
        x = self.conv_bn_block_77(x)
        x = self.conv_bn_block_78(x)
        conv_79 = self.conv_bn_block_79(x)
        conv_80 = self.conv_bn_block_80(conv_79)
        conv_81 = self.conv_bn_block_81(conv_80)

        predict_82 = self.transform_0(conv_81)

        route_83 = conv_79
        conv_84 = self.conv_bn_block_84(route_83)
        upsample_85 = self.upsample_block_85(conv_84)
        route_86 = F.concat(upsample_85, shortcut_61, dim=1)

        x = self.conv_bn_block_87(route_86)
        x = self.conv_bn_block_88(x)
        x = self.conv_bn_block_89(x)
        x = self.conv_bn_block_90(x)
        conv_91 = self.conv_bn_block_91(x)
        x = self.conv_bn_block_92(conv_91)
        conv_93 = self.conv_bn_block_93(x)

        predict_94 = self.transform_1(conv_93)

        route_95 = conv_91
        conv_96 = self.conv_bn_block_96(route_95)
        upsample_97 = self.upsample_block_97(conv_96)
        route_98 = F.concat(upsample_97, shortcut_36, dim=1)

        x = self.conv_bn_block_99(route_98)
        x = self.conv_bn_block_100(x)
        x = self.conv_bn_block_101(x)
        x = self.conv_bn_block_102(x)
        x = self.conv_bn_block_103(x)
        x = self.conv_bn_block_104(x)
        conv_105 = self.conv_bn_block_105(x)

        predict_106 = self.transform_2(conv_105)
        detections = F.concat(predict_82, predict_94, predict_106, dim=1)

        return detections

    def load_weights(self, weightfile, fine_tune):
        # Open the weights file
        fp = open(weightfile, "rb")

        # The first 5 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4,5. Images seen by the network (during training)
        self.header = nd.array(np.fromfile(fp, dtype=np.int32, count=5))
        self.seen = self.header[3]

        weights = nd.array(np.fromfile(fp, dtype=np.float32))
        ptr = 0

        def set_data(model, ptr):
            conv = model[0]
            if len(model) > 1:
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
            else:
                num_biases = self.numel(conv.bias.shape)

                conv_biases = weights[ptr: ptr + num_biases]
                ptr = ptr + num_biases

                conv_biases = conv_biases.reshape(conv.bias.shape)

                conv.bias.set_data(conv_biases)

            num_weights = self.numel(conv.weight.shape)

            conv_weights = weights[ptr:ptr + num_weights]
            ptr = ptr + num_weights

            conv_weights = conv_weights.reshape(conv.weight.shape)

            conv.weight.set_data(conv_weights)
            return ptr

        modules = self._children
        for block_name in modules:
            if fine_tune:
                if block_name.find("81") != -1:
                    ptr = 56629087
                    continue
                elif block_name.find("93") != -1:
                    ptr = 60898910
                    continue
                elif block_name.find("105") != -1:
                    continue
            module = modules.get(block_name)
            if isinstance(module, nn.HybridSequential):
                ptr = set_data(module, ptr)
            elif isinstance(module, ShortCutBlock):
                shortcut_modules = module._children
                for shortcut_name in shortcut_modules:
                    ptr = set_data(shortcut_modules.get(shortcut_name), ptr)
            elif isinstance(module, UpSampleBlock) or isinstance(module, TransformBlock):
                continue
            else:
                print(module)
                print("load weights wrong")

    def numel(self, x):
        if isinstance(x, nd.NDArray):
            x = x.asnumpy()
        return np.prod(x)


class TinyDarkNet(nn.HybridBlock):
    def __init__(self, num_classes=80, input_dim=416):
        super(TinyDarkNet, self).__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.anchors = [(10, 14), (23, 27), (37, 58), (81, 82), (135, 169), (344, 319)]

        self.conv_bn_block_0 = ConvBNBlock(16, 3, 1, 1)
        self.max_pool_1 = nn.MaxPool2D(2, 2)
        self.conv_bn_block_2 = ConvBNBlock(32, 3, 1, 1)
        self.max_pool_3 = nn.MaxPool2D(2, 2)
        self.conv_bn_block_4 = ConvBNBlock(64, 3, 1, 1)
        self.max_pool_5 = nn.MaxPool2D(2, 2)
        self.conv_bn_block_6 = ConvBNBlock(128, 3, 1, 1)
        self.max_pool_7 = nn.MaxPool2D(2, 2)
        self.conv_bn_block_8 = ConvBNBlock(256, 3, 1, 1)
        self.max_pool_9 = nn.MaxPool2D(2, 2)
        self.conv_bn_block_10 = ConvBNBlock(512, 3, 1, 1)
        self.max_pool_11 = nn.MaxPool2D(2, 1, same_mode=True)
        self.conv_bn_block_12 = ConvBNBlock(1024, 3, 1, 1)
        self.conv_bn_block_13 = ConvBNBlock(256, 1, 1, 1)
        self.conv_bn_block_14 = ConvBNBlock(512, 3, 1, 1)
        self.conv_bn_block_15 = ConvBNBlock(3 * (5 + self.num_classes), 1, 1, 1, use_bias=True, leaky=False)
        self.transform_0 = TransformBlock(self.num_classes, 13)
        self.conv_bn_block_16 = ConvBNBlock(128, 1, 1, 1)
        self.upsample_block_17 = UpSampleBlock(scale=2)
        self.conv_bn_block_18 = ConvBNBlock(256, 3, 1, 1)
        self.conv_bn_block_19 = ConvBNBlock(3 * (5 + self.num_classes), 1, 1, 1, use_bias=True, leaky=False)
        self.transform_1 = TransformBlock(self.num_classes, 26)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.conv_bn_block_0(x)
        x = self.max_pool_1(x)
        x = self.conv_bn_block_2(x)
        x = self.max_pool_3(x)
        x = self.conv_bn_block_4(x)
        x = self.max_pool_5(x)
        x = self.conv_bn_block_6(x)
        x = self.max_pool_7(x)
        conv_8 = self.conv_bn_block_8(x)
        x = self.max_pool_9(conv_8)
        x = self.conv_bn_block_10(x)
        x = self.max_pool_11(x)
        x = self.conv_bn_block_12(x)
        conv_13 = self.conv_bn_block_13(x)
        x = self.conv_bn_block_14(conv_13)
        conv_15 = self.conv_bn_block_15(x)

        predict_16 = self.transform_0(conv_15)

        x = self.conv_bn_block_16(conv_13.copy())
        x = self.upsample_block_17(x)
        x = nd.concat(x, conv_8, dim=1)
        x = self.conv_bn_block_18(x)
        x = self.conv_bn_block_19(x)
        predict_20 = self.transform_1(x)
        detections = nd.concat(predict_16, predict_20, dim=1)

        return detections

    def load_weights(self, weightfile, fine_tune):
        # Open the weights file
        fp = open(weightfile, "rb")

        # The first 5 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4,5. Images seen by the network (during training)
        self.header = nd.array(np.fromfile(fp, dtype=np.int32, count=5))
        self.seen = self.header[3]

        weights = nd.array(np.fromfile(fp, dtype=np.float32))
        ptr = 0

        def set_data(model, ptr):
            conv = model[0]
            if len(model) > 1:
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
            else:
                num_biases = self.numel(conv.bias.shape)

                conv_biases = weights[ptr: ptr + num_biases]
                ptr = ptr + num_biases

                conv_biases = conv_biases.reshape(conv.bias.shape)

                conv.bias.set_data(conv_biases)

            num_weights = self.numel(conv.weight.shape)

            conv_weights = weights[ptr:ptr + num_weights]
            ptr = ptr + num_weights

            conv_weights = conv_weights.reshape(conv.weight.shape)

            conv.weight.set_data(conv_weights)
            return ptr

        modules = self._children
        for block_name in modules:
            if fine_tune:
                if block_name.find("81") != -1:
                    ptr = 56629087
                    continue
                elif block_name.find("93") != -1:
                    ptr = 60898910
                    continue
                elif block_name.find("105") != -1:
                    continue
            module = modules.get(block_name)
            if isinstance(module, nn.HybridSequential):
                ptr = set_data(module, ptr)
            elif isinstance(module, UpSampleBlock) or isinstance(module, TransformBlock) or isinstance(module, nn.MaxPool2D):
                continue
            else:
                print(module)
                print("load weights wrong")

    def numel(self, x):
        if isinstance(x, nd.NDArray):
            x = x.asnumpy()
        return np.prod(x)


if __name__ == '__main__':
    darknet = DarkNet()
    darknet.initialize()
    X = nd.uniform(shape=(1, 3, 416, 416))
    detections = darknet(X)
    darknet.load_weights("yolov3.weights")
    print(detections.shape)
