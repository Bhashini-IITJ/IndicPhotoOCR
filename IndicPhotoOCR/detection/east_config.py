
# data-config
import numpy as np

train_data_path = './dataset/train/'
train_batch_size_per_gpu = 14  # 14
num_workers = 24  # 24
gpu_ids = [0]  # [0,1,2,3]
gpu = 1  # 4
input_size = 512  # 预处理后归一化后图像尺寸
background_ratio = 3. / 8  # 纯背景样本比例
random_scale = np.array([0.5, 1, 2.0, 3.0])  # 提取多尺度图片信息
geometry = 'RBOX'  # 选择使用几何特征图类型
max_image_large_side = 1280
max_text_size = 800
min_text_size = 10
min_crop_side_ratio = 0.1
means=[100, 100, 100]
pretrained = True  # 是否加载基础网络的预训练模型
pretrained_basemodel_path = 'IndicPhotoOCR/detection/East/tmp/backbone_net/mobilenet_v2.pth.tar'
pre_lr = 1e-4  # 基础网络的初始学习率
lr = 1e-3  # 后面网络的初始学习率
decay_steps = 50  # decayed_learning_rate = learning_rate * decay_rate ^ (global_epoch / decay_steps)
decay_rate = 0.97
init_type = 'xavier'  # 网络参数初始化方式
resume = True  # 整体网络是否恢复原来保存的模型
checkpoint = 'IndicPhotoOCR/detection/East/tmp/epoch_990_checkpoint.pth.tar'  # 指定具体路径及文件名
max_epochs = 1000  # 最大迭代epochs数
l2_weight_decay = 1e-6  # l2正则化惩罚项权重
print_freq = 10  # 每10个batch输出损失结果
save_eval_iteration = 50  # 每10个epoch保存一次模型,并做一次评价
save_model_path = './tmp/'  # 模型保存路径
test_img_path = './dataset/full_set'  # demo测试样本路径'./demo/test_img/'，数据集测试为'./dataset/test/'
res_img_path = 'results'  # demo结果存放路径'./demo/result_img/'，数据集测试为 './dataset/test_result/'
write_images = True  # 是否输出图像结果
score_map_thresh = 0.8  # 置信度阈值
box_thresh = 0.1  # 文本框中置信度平均值的阈值
nms_thres = 0.2  # 局部非极大抑制IOU阈值
compute_hmean_path = './dataset/test_compute_hmean/'
