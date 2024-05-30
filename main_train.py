import argparse
from trainer_exchange import trainer
import torch
try:
    from itertools import izip as zip
except ImportError:
    pass
import sys
from random import shuffle
from torchvision import transforms
from data import ImageFolder
from torch.utils.data import DataLoader

import torchvision.utils as vutils
import os


parser = argparse.ArgumentParser()
parser.add_argument('--cuda',type=str,default='True',help='Use gpu or not')
parser.add_argument('--model_save_path',type=str,default='models/')
parser.add_argument('--image_save_path',type=str,default='images_display/',help='display of generative cases')
parser.add_argument('--trainA_img_path',type=str,default='trainA_img/') # training images with class A are placed into this folder
parser.add_argument('--trainB_img_path',type=str,default='trainB_img/') # training images with class B are placed into this folder
parser.add_argument('--testA_img_path',type=str,default='testA_img/') # testing images with class A are placed into this folder
parser.add_argument('--testB_img_path',type=str,default='testB_img/') # testing images with class B are placed into this folder
parser.add_argument('--train_label_file',type=str,default='trainAB_img-name_label.txt') # name and label of each training image are presented in each line in this file, like the first line:"brain_image_name1 1", where 1 refers to the abnormal class.
parser.add_argument('--test_label_file',type=str,default='testAB_img-name_label.txt')
parser.add_argument('--batch_size',type=int,default=1)
parser.add_argument('--num_workers',type=int,default=4)
parser.add_argument('--display_size',type=int,default=16)
parser.add_argument('--model_save_iter',type=int,default=2000)
parser.add_argument('--image_save_iter',type=int,default=2000)
parser.add_argument('--train_max_iter',type=int,default=1000000)

opts = parser.parse_args()
if opts.cuda=='True':
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

trainer = trainer(device=device)
trainer.to(device)


# image path
trainA_img_path=opts.trainA_img_path
trainB_img_path=opts.trainB_img_path
testA_img_path=opts.testA_img_path
testB_img_path=opts.testB_img_path

# label path
train_label_file=opts.train_label_file
test_label_file=opts.test_label_file


# data loader
transform1=transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(256),
    transforms.CenterCrop((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform2=transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset_trainA=ImageFolder(trainA_img_path, train_label_file,transform=transform1)
dataset_trainB=ImageFolder(trainB_img_path, train_label_file,transform=transform1)
dataset_testA=ImageFolder(testA_img_path, test_label_file,transform=transform2)
dataset_testB=ImageFolder(testB_img_path, test_label_file,transform=transform2)
batch_size=opts.batch_size
num_workers=opts.num_workers
train_loader_a=DataLoader(dataset=dataset_trainA, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
train_loader_b=DataLoader(dataset=dataset_trainB, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
test_loader_a=DataLoader(dataset=dataset_testA, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers)
test_loader_b =DataLoader(dataset=dataset_testB, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers)


## display set
display_size=opts.display_size
def __write_images(image_outputs, display_image_num, file_name):
    image_outputs = [images.expand(-1, 3, -1, -1) for images in image_outputs] # expand gray-scale images to 3 channels
    image_tensor = torch.cat([images[:display_image_num] for images in image_outputs], 0)
    image_grid = vutils.make_grid(image_tensor.data, nrow=display_image_num, padding=0, normalize=True)
    vutils.save_image(image_grid, file_name, nrow=1)
def write_2images(image_outputs, display_image_num, image_directory, postfix):
    n = len(image_outputs)
    __write_images(image_outputs[0:n//2], display_image_num, '%s/gen_a2b_%s.jpg' % (image_directory, postfix))
    __write_images(image_outputs[n//2:n], display_image_num, '%s/gen_b2a_%s.jpg' % (image_directory, postfix))
# train
ta_list=[]
for i in range(len(train_loader_a.dataset)):
    ta_list.append(i)
tb_list=[]
for i in range(len(train_loader_b.dataset)):
    tb_list.append(i)
shuffle(ta_list)
train_display_images_a = torch.stack([train_loader_a.dataset[i][0] for i in ta_list[:display_size]]).to(device)
shuffle(tb_list)
train_display_images_b1 = torch.stack([train_loader_b.dataset[i][0] for i in tb_list[:display_size]]).to(device)
shuffle(tb_list)
train_display_images_b2 = torch.stack([train_loader_b.dataset[i][0] for i in tb_list[:display_size]]).to(device)
shuffle(tb_list)
train_display_images_b3 = torch.stack([train_loader_b.dataset[i][0] for i in tb_list[:display_size]]).to(device)
shuffle(tb_list)
train_display_images_b4 = torch.stack([train_loader_b.dataset[i][0] for i in tb_list[:display_size]]).to(device)
shuffle(tb_list)
train_display_images_b5 = torch.stack([train_loader_b.dataset[i][0] for i in tb_list[:display_size]]).to(device)
# test
ea_list=[]
for i in range(len(test_loader_a.dataset)):
    ea_list.append(i)
eb_list=[]
for i in range(len(test_loader_b.dataset)):
    eb_list.append(i)
shuffle(ea_list)
test_display_images_a = torch.stack([test_loader_a.dataset[i][0] for i in ea_list[:display_size]]).to(device)
shuffle(eb_list)
test_display_images_b1 = torch.stack([test_loader_b.dataset[i][0] for i in eb_list[:display_size]]).to(device)
shuffle(eb_list)
test_display_images_b2 = torch.stack([test_loader_b.dataset[i][0] for i in eb_list[:display_size]]).to(device)
shuffle(eb_list)
test_display_images_b3 = torch.stack([test_loader_b.dataset[i][0] for i in eb_list[:display_size]]).to(device)
shuffle(eb_list)
test_display_images_b4 = torch.stack([test_loader_b.dataset[i][0] for i in eb_list[:display_size]]).to(device)
shuffle(eb_list)
test_display_images_b5 = torch.stack([test_loader_b.dataset[i][0] for i in eb_list[:display_size]]).to(device)



if __name__=='__main__':
    # Start training
    iterations = 0
    image_save_iter=opts.image_save_iter
    model_save_iter=opts.model_save_iter
    image_save_path=opts.image_save_path
    model_save_path = opts.model_save_path
    max_iter=opts.train_max_iter
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    while True:
        for it, (images_label_a, images_label_b) in enumerate(zip(train_loader_a, train_loader_b)):

            ##image a and b with label
            images_a=images_label_a[0]
            a_label=images_label_a[1]
            images_b=images_label_b[0]
            b_label=images_label_b[1]
            images_a, images_b = images_a.to(device).detach(), images_b.to(device).detach()
            a_label,b_label=a_label.to(device),b_label.to(device)

            # discriminator update
            trainer.dis_update(images_a, images_b,dis_a_real_label=a_label,dis_b_real_label=b_label)
            # encoder and decoder update
            trainer.gen_update(images_a, images_b,dis_a_real_label=a_label,dis_b_real_label=b_label)
            #learning rate update
            trainer.update_learning_rate()

            # save display images
            if (iterations + 1) % image_save_iter == 0:
                with torch.no_grad():
                    test_image_outputs1 = trainer.sample(test_display_images_a, test_display_images_b1)
                    train_image_outputs1 = trainer.sample(train_display_images_a, train_display_images_b1)
                    test_image_outputs2 = trainer.sample(test_display_images_a, test_display_images_b2)
                    train_image_outputs2 = trainer.sample(train_display_images_a, train_display_images_b2)
                    test_image_outputs3 = trainer.sample(test_display_images_a, test_display_images_b3)
                    train_image_outputs3 = trainer.sample(train_display_images_a, train_display_images_b3)
                    test_image_outputs4 = trainer.sample(test_display_images_a, test_display_images_b4)
                    train_image_outputs4 = trainer.sample(train_display_images_a, train_display_images_b4)
                    test_image_outputs5 = trainer.sample(test_display_images_a, test_display_images_b5)
                    train_image_outputs5 = trainer.sample(train_display_images_a, train_display_images_b5)

                write_2images(test_image_outputs1, display_size, image_save_path, 'test1_%08d' % (iterations + 1))
                write_2images(train_image_outputs1, display_size, image_save_path, 'train1_%08d' % (iterations + 1))
                write_2images(test_image_outputs2, display_size, image_save_path, 'test2_%08d' % (iterations + 1))
                write_2images(train_image_outputs2, display_size, image_save_path, 'train2_%08d' % (iterations + 1))
                write_2images(test_image_outputs3, display_size, image_save_path, 'test3_%08d' % (iterations + 1))
                write_2images(train_image_outputs3, display_size, image_save_path, 'train3_%08d' % (iterations + 1))
                write_2images(test_image_outputs4, display_size, image_save_path, 'test4_%08d' % (iterations + 1))
                write_2images(train_image_outputs4, display_size, image_save_path, 'train4_%08d' % (iterations + 1))
                write_2images(test_image_outputs5, display_size, image_save_path, 'test5_%08d' % (iterations + 1))
                write_2images(train_image_outputs5, display_size, image_save_path, 'train5_%08d' % (iterations + 1))

            # Save model
            if (iterations + 1) % model_save_iter == 0:
                trainer.save(model_save_path, iterations)

            print(it)

            iterations += 1
            if iterations >= max_iter:
                sys.exit('Finish training')
