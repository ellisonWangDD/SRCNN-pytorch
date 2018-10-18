import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader import SRCNN_dataset
from model import SRCNN
import time
import os

#==================================================================

class AverageMeter():
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.sum = 0
        self.avg = 0
        self.count = 0
    def update(self, loss, n=1):
        self.count += n
        self.val = loss
        self.sum += loss * n
        self.avg = self.sum / self.count

def train(data_set, model, loss_fn, optimizer, epoch, print_interval=2, use_cuda=True, Writer=None):
    losses = AverageMeter()#loss使用averageMeter进行准确测量
    # time = AverageMeter()
    for step, (input_sample, label_sample) in enumerate(train_loader):
        start = time.time()
        if use_cuda:
            input_sample, label_sample = input_sample.cuda(), label_sample.cuda()#transform tensor to CUDA,不需要置require_grad=True
        output = model(input_sample)
        loss = loss_fn(output, label_sample)
        optimizer.zero_grad()
        loss.backward()
        losses.update(loss.item(), input_sample.size(0))
        optimizer.step()
        if step % print_interval == 0:
            print('========>Epoch:{}  Step:{} Loss:{:.4f}({:.4f}) Consume_time:{:.5f}=========='.format(epoch, step, loss.item(), losses.avg, time.time()-start))
    if not Writer:#此处使用try...except进行管理
        try:
            Writer.add_scalar('training_loss', losses.avg, epoch)
        except NameError as reason:
            print('You have not initilize Writer', reason)
    return losses.avg

def validate(img_path,model, pic_mode='.bmp'):
    import numpy as np
    from PIL import Image
    import cv2
    img = np.array(Image.open(img_path).convert('L'))#transform to Gray image
    assert isinstance(img, np.ndarray)#判断是否是np.ndarray实例,提前预警报错
    img = img.reshape(1, 1, img.shape[0], img.shape[1])#对shape进行调整,必须是4个dimension
    img = torch.FloatTensor(img).cuda()#传入网络之前必须先转换为cuda类型
    out = model(img)#网络的输出实际上是Variable
    out = out.squeeze().cpu().data.numpy()#先将所有数据加载到CPU才可以正常使用.输出数据是Variable类型,需要.data转换为tensor
    save_name = 'out' + pic_mode
    cv2.imwrite(save_name, out)
    print('-----------save to out.jpg--------------')

#================================================
train_config = {
    'dir_path': 'test/testset/test2',
    'scale': 3,
    'is_gray': True,
    'input_size': 33,
    'label_size': 21,
    'stride': 21,
    'lr':1e-4,
    'batch_size':2000
}
#================================================

test_config = train_config.copy()
use_cuda = True
use_tensorBoard = True
resume = True
checkpiont_path = 'model/E95L10000.pkl'
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
Start_epoch = 0
End_epoch = 100
Evaluation = True
test_path = '3.bmp'
#================================================

model = SRCNN()

if use_cuda:
    model = nn.DataParallel(model).cuda()#不要忘记在Dataparallel后面也加上.cuda()

if resume:
    checkpoint = torch.load(checkpiont_path)#如果之前使用了model=dataparallel之后,网络结构会有所变化,因此在读取之前同样要赋值操作
    model.load_state_dict(checkpoint['state'])
    Start_epoch = checkpoint['epoch']
    train_config['lr'] = checkpoint['lr']
    print('===============Load Model Successully', checkpiont_path)

try:
    from tensorboardX import SummaryWriter
except ImportError as reason:
    print('Use pip install tensorboardX', reason)
else:
    writer = SummaryWriter(log_dir='writer_log', comment='--')
    print('================Use TensorBoard ================')

#=================================================

if Evaluation:
    validate(test_path, model)
else:
    train_dataset = SRCNN_dataset(train_config)
    criterion = nn.MSELoss().cuda()
    optimizer_adam = optim.Adam(model.parameters(), lr=train_config['lr'])
    train_dataset = SRCNN_dataset(train_config)
    train_loader = DataLoader(dataset=train_dataset, batch_size=train_config['batch_size'])
    #==========================================================================================
    for _epoch in range(Start_epoch, End_epoch):
        loss_avg = train(train_loader, model, criterion, optimizer_adam, _epoch, Writer=writer)#将train过程封装成函数,这样使整体代码结构清晰
        save_state = {'epoch': _epoch,#存储网络的时候不要只存储state_dict(),要把一些关键参数都存进去
                  'lr': train_config['lr'],
                  'state': model.state_dict()}
        if _epoch % 5 == 0:
            if not os.path.exists('model/'):#标准步骤,检测文件夹是否存在
                os.mkdir('model/')#不存在进行创建
            save_name = 'model/E%dL%d.pkl'%(_epoch, int(10000 *loss_avg))
            torch.save(save_state, save_name)


def save_checkpoint(state, file_path='model/%filename'):
    torch.save(state, file_path)