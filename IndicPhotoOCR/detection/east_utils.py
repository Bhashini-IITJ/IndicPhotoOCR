
import torch
import os
from torch.nn import init
import cv2
import numpy as np
import time
import requests

from IndicPhotoOCR.detection import east_config as cfg
from IndicPhotoOCR.detection import east_preprossing as preprossing
from IndicPhotoOCR.detection import east_locality_aware_nms as locality_aware_nms



# Example usage:
model_info = {
    "east": {
        "paths": [ cfg.checkpoint, cfg.pretrained_basemodel_path],
        "urls" : ["https://github.com/anikde/STocr/releases/download/e0.1.0/epoch_990_checkpoint.pth.tar", "https://github.com/anikde/STocr/releases/download/e0.1.0/mobilenet_v2.pth.tar"]
    },
}

class ModelManager:
    def __init__(self):
        # self.root_model_dir = "bharatOCR/detection/"
        pass

    def download_model(self, url, path):
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # Filter out keep-alive chunks
                        f.write(chunk)
            print(f"Downloaded: {path}")
        else:
            print(f"Failed to download from {url}")

    def ensure_model(self, model_name):
        model_paths = model_info[model_name]["paths"]  # Changed to handle multiple paths
        urls = model_info[model_name]["urls"]          # Changed to handle multiple URLs


        for model_path, url in zip(model_paths, urls):
            # full_model_path = os.path.join(self.root_model_dir, model_path)

            # Ensure the model path directory exists
            os.makedirs(os.path.dirname(os.path.join(*cfg.pretrained_basemodel_path.split("/"))), exist_ok=True)

            if not os.path.exists(model_path):
                print(f"Model not found locally. Downloading {model_name} from {url}...")
                self.download_model(url, model_path)
            else:
                print(f"Model already exists at {model_path}. No need to download.")



# # Initialize ModelManager and ensure Hindi models are downloaded
model_manager = ModelManager()
model_manager.ensure_model("east")



def init_weights(m_list, init_type=cfg.init_type, gain=0.02):
    print("EAST <==> Prepare <==> Init Network'{}' <==> Begin".format(cfg.init_type))
    # this will apply to each layer
    for m in m_list:
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')  # good for relu
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print("EAST <==> Prepare <==> Init Network'{}' <==> Done".format(cfg.init_type))


def Loading_checkpoint(model, optimizer, scheduler, filename='checkpoint.pth.tar'):
    """[summary]
    [description]
    Arguments:
        state {[type]} -- [description] a dict describe some params
    Keyword Arguments:
        filename {str} -- [description] (default: {'checkpoint.pth.tar'})
    """
    weightpath = os.path.abspath(cfg.checkpoint)
    print("EAST <==> Prepare <==> Loading checkpoint '{}' <==> Begin".format(weightpath))
    checkpoint = torch.load(weightpath)
    start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    print("EAST <==> Prepare <==> Loading checkpoint '{}' <==> Done".format(weightpath))

    return start_epoch


def save_checkpoint(epoch, model, optimizer, scheduler, filename='checkpoint.pth.tar'):
    """[summary]
    [description]
    Arguments:
        state {[type]} -- [description] a dict describe some params
    Keyword Arguments:
        filename {str} -- [description] (default: {'checkpoint.pth.tar'})
    """
    print('EAST <==> Save weight - epoch {} <==> Begin'.format(epoch))
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    weight_dir = cfg.save_model_path
    if not os.path.exists(weight_dir):
        os.mkdir(weight_dir)
    filename = 'epoch_' + str(epoch) + '_checkpoint.pth.tar'
    file_path = os.path.join(weight_dir, filename)
    torch.save(state, file_path)
    print('EAST <==> Save weight - epoch {} <==> Done'.format(epoch))


class Regularization(torch.nn.Module):
    def __init__(self, model, weight_decay, p=2):
        super(Regularization, self).__init__()
        if weight_decay < 0:
            print("param weight_decay can not <0")
            exit(0)
        self.model = model
        self.weight_decay = weight_decay
        self.p = p
        self.weight_list = self.get_weight(model)
        # self.weight_info(self.weight_list)

    def to(self, device):
        self.device = device
        super().to(device)
        return self

    def forward(self, model):
        self.weight_list = self.get_weight(model)  # 获得最新的权重
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss

    def get_weight(self, model):
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_list, weight_decay, p=2):
        reg_loss = 0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg

        reg_loss = weight_decay * reg_loss
        return reg_loss

    def weight_info(self, weight_list):
        print("---------------regularization weight---------------")
        for name, w in weight_list:
            print(name)
        print("---------------------------------------------------")


def resize_image(im, max_side_len=2400):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    """
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.

    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)
    """

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    #resize_h, resize_w = 512, 512
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def detect(score_map, geo_map, timer, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''

    # score_map 和 geo_map 的维数进行调整
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, :]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = preprossing.restore_rectangle(xy_text[:, ::-1] * 4,
                                                      geo_map[xy_text[:, 0], xy_text[:, 1], :])  # N*4*2
    # print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start
    # nms part
    start = time.time()
    boxes = locality_aware_nms.nms_locality(boxes.astype(np.float64), nms_thres)
    timer['nms'] = time.time() - start
    # print(timer['nms'])
    if boxes.shape[0] == 0:
        return None, timer

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]
    return boxes, timer


def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis + 1) % 4, (min_axis + 2) % 4, (min_axis + 3) % 4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]


def mean_image_subtraction(images, means=cfg.means):
    '''
    image normalization
    :param images: bs * w * h * channel
    :param means:
    :return:
    '''
    num_channels = images.data.shape[1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    for i in range(num_channels):
        images.data[:, i, :, :] -= means[i]

    return images
