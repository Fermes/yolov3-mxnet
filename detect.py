import os
import time
import argparse
from util_mxnet import *
from darknet_mxnet import DarkNet

image_name = 0


def arg_parse():
    parser = argparse.ArgumentParser(description="YOLO v3 Detection Module")
    parser.add_argument("--images", dest='images', help=
    "Image / Directory containing images to perform detection upon",
                        default="images", type=str)
    parser.add_argument("--video", dest='video', help=
    "video file path", type=str)
    parser.add_argument("--classes", dest="classes", default="data/coco.names", type=str)
    parser.add_argument("--gpu", dest="gpu", help="gpu id", default=0, type=int)
    parser.add_argument("--dst_dir", dest='dst_dir', help=
    "Image / Directory to store detections to",
                        default="results", type=str)
    parser.add_argument("--batch_size", dest="batch_size", help="Batch size", default=16, type=int)
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence", default=0.5, type=float)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4, type=float)
    parser.add_argument("--params", dest='params', help=
    "params file", type=str)
    parser.add_argument("--input_dim", dest='input_dim', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default=416, type=int)

    return parser.parse_args()


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


def draw_bbox(img, bboxs):
    for x in bboxs:
        c1 = tuple(x[1:3].astype(np.int))
        c2 = tuple(x[3:5].astype(np.int))
        cls = int(x[-1])
        # color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        color = (255, 0, 0)
        label = "{0} {1:.3f}".format(classes[cls], x[-2])
        cv2.rectangle(img, c1, c2, color, 2)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c2 = c1[0] + t_size[0] + 2, c1[1] - t_size[1] - 5
        cv2.rectangle(img, c1, c2, color, -1)
        cv2.putText(img, label, (c1[0], c1[1] - t_size[1] + 7), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
    return img


def save_results(load_images, output, input_dim):
    im_dim_list = nd.array([(x.shape[1], x.shape[0]) for x in load_images])
    im_dim_list = nd.tile(im_dim_list, 2)
    im_dim_list = im_dim_list[output[:, 0], :]

    scaling_factor = input_dim / im_dim_list
    # scaling_factor = (416 / im_dim_list)[0].view(-1, 1)

    output[:, [1, 3]] -= (input_dim - scaling_factor[:, 0:1] * im_dim_list[:, 0:1].reshape((-1, 1))) / 2
    output[:, [2, 4]] -= (input_dim - scaling_factor[:, 1:2] * im_dim_list[:, 1:2].reshape((-1, 1))) / 2
    output[:, 1:5] /= scaling_factor

    for i in range(output.shape[0]):
        output[i, [1, 3]] = nd.clip(output[i, [1, 3]], a_min=0.0, a_max=im_dim_list[i, 0].asscalar())
        output[i, [2, 4]] = nd.clip(output[i, [2, 4]], a_min=0.0, a_max=im_dim_list[i, 1].asscalar())

    output = output.asnumpy()
    for i in range(len(load_images)):
        bboxs = []
        for bbox in output:
            if i == int(bbox[0]):
                bboxs.append(bbox)
        draw_bbox(load_images[i], bboxs)
    global image_name
    list(map(cv2.imwrite, [os.path.join(dst_dir, "{0}.jpg".format(image_name + i)) for i in range(len(load_images))], load_images))
    image_name += len(load_images)


def predict_video(net, ctx, video_file):
    if video_file:
        cap = cv2.VideoCapture(video_file)
    else:
        cap = cv2.VideoCapture(0)

    assert cap.isOpened(), 'Cannot capture source'

    frames = 0
    start = time.time()
    result_video = cv2.VideoWriter(
        os.path.join(dst_dir, "result.avi"),
        cv2.VideoWriter_fourcc("X", "2", "6", "4"),
        37,
        (1280, 720)
    )
    anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                       (59, 119), (116, 90), (156, 198), (373, 326)])
    cost_time = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cost_start = time.time()
            frames += 1
            # if frames % 2 != 0:
            #     continue
            # frame = cv2.resize(frame, (1280, 720))
            img = nd.array(prep_image(frame, input_dim), ctx=ctx).expand_dims(0)
            # cv2.imshow("a", frame)

            prediction = predict_transform(net(img), input_dim, anchors)
            prediction = write_results(prediction, num_classes, confidence=confidence, nms_conf=nms_thresh)

            if type(prediction) == int:
                frames += 1
                print("FPS of the video is {:5.4f}".format(frames / (time.time() - start)))
                result_video.write(frame)
                # cv2.imshow("frame", frame)
                # key = cv2.waitKey(1)
                # if key & 0xFF == ord('q'):
                #     break
                continue
            # output[:, 1:5] = torch.clamp(output[:, 1:5], 0.0, float(input_dim))

            # im_dim = im_dim.repeat(output.size(0), 1) / input_dim
            # output[:, 1:5] *= im_dim
            prediction = prediction.asnumpy()
            scaling_factor = min(input_dim / frame.shape[0], input_dim / frame.shape[1])
            # scaling_factor = (416 / im_dim_list)[0].view(-1, 1)

            prediction[:, [1, 3]] -= (input_dim - scaling_factor * frame.shape[1]) / 2
            prediction[:, [2, 4]] -= (input_dim - scaling_factor * frame.shape[0]) / 2
            prediction[:, 1:5] /= scaling_factor

            for i in range(prediction.shape[0]):
                prediction[i, [1, 3]] = np.clip(prediction[i, [1, 3]], 0.0, frame.shape[1])
                prediction[i, [2, 4]] = np.clip(prediction[i, [2, 4]], 0.0, frame.shape[0])
            draw_bbox(frame, prediction)

            cost_time += time.time() - cost_start
            result_video.write(frame)

            # cv2.imshow("frame", frame)
            # key = cv2.waitKey(1000)
            # if key & 0xFF == ord('q'):
            #     break
            # frames += 1
            # print(time.time() - start)
            if frames % 37 == 0:
                print("FPS of the video is {:5.2f}\n Per Image Cost Time{:5.2f}".format(frames / (time.time() - start),
                                                                                        cost_time / 37))
                cost_time = 0
        else:
            print("video source closed")
            break
    result_video.release()
    print("{0} detect complete".format(video_file))


if __name__ == '__main__':
    args = arg_parse()
    images = args.images
    batch_size = args.batch_size
    confidence = args.confidence
    nms_thresh = args.nms_thresh
    input_dim = args.input_dim
    dst_dir = args.dst_dir
    start = 0
    # classes = load_classes("data/coco.names")
    classes = load_classes(args.classes)

    ctx = try_gpu(args.gpu)[0]
    num_classes = len(classes)
    net = DarkNet(input_dim=input_dim, num_classes=num_classes)
    net.initialize(ctx=ctx)
    input_dim = args.input_dim

    try:
        imlist = [os.path.join(images, img) for img in os.listdir(images)]
    except NotADirectoryError:
        imlist = []
    except FileNotFoundError:
        print("No file or directory with the name {}".format(images))

    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    tmp_batch = nd.uniform(shape=(1, 3, args.input_dim, args.input_dim), ctx=ctx)
    net(tmp_batch)
    if args.params:
        net.load_params(args.params)
    else:
        net.load_weights("./data/yolov3.weights", fine_tune=False)

    if args.video:
        predict_video(net, ctx=ctx, video_file=args.video)
        exit()

    if not imlist:
        print("no images to detect")
        exit()
    leftover = 0
    if len(imlist) % batch_size:
        leftover = 1

    num_batches = len(imlist) // batch_size + leftover
    im_batches = [imlist[i * batch_size: min((i + 1) * batch_size, len(imlist))]
                  for i in range(num_batches)]

    start_det_loop = time.time()
    anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                       (59, 119), (116, 90), (156, 198), (373, 326)])
    output = None
    for i, batch in enumerate(im_batches):
        load_images = [cv2.imread(img) for img in batch]
        tmp_batch = list(map(prep_image, load_images, [input_dim for x in range(len(batch))]))
        tmp_batch = nd.array(tmp_batch, ctx=ctx)
        start = time.time()
        prediction = predict_transform(net(tmp_batch), input_dim, anchors)
        # from train import prep_label
        # label = prep_label("./data/000283.txt", num_classes, ctx=prediction.context)
        # prediction, _ = prep_final_label(label.expand_dims(0), num_classes, ctx=prediction.context, input_dim=input_dim)
        # prediction = predict_transform(prediction, input_dim, anchors)
        prediction = write_results(prediction, num_classes, confidence=confidence, nms_conf=nms_thresh)

        end = time.time()

        if type(prediction) == int:
            for im_num, image in enumerate(imlist[i * batch_size: min((i + 1) * batch_size, len(imlist))]):
                im_id = i * batch_size + im_num
                print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start) / batch_size))
                print("----------------------------------------------------------")
            continue

        # prediction[:, 0] += i * batch_size  # transform the atribute from index in batch to index in imlist

        if output is None:  # If we have't initialised output
            output = prediction
        else:
            output = nd.concat(output, prediction, dim=0)

        for image in batch:
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start) / len(batch)))
            print("----------------------------------------------------------")

        if not output is None:
            save_results(load_images, output, input_dim=input_dim)
        else:
            print("No detections were made")
        output = None
