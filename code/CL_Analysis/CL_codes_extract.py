import os
import torch
from torch.autograd import Variable
from PIL import Image
from trainer_exchange import trainer
from torchvision import transforms
import argparse
import csv


parser = argparse.ArgumentParser()
parser.add_argument('--cuda',type=str,default='True',help='Use gpu or not')
parser.add_argument('--A_img_path',type=str,default='/data/Brain Tumor2/testA_img/') ###testing images with class A are placed into this folder
parser.add_argument('--B_img_path',type=str,default='/data/Brain Tumor2/testB_img/') ###testing images with class B are placed into this folder
parser.add_argument('--AB_image_name_label_path',type=str,default='/data/Brain Tumor2/testAB_img-name_label.txt')
parser.add_argument('--latentAB_save_path',type=str,default='/results/testAB_CL_codes_extraction_results.csv')  ###where the class-associated codes extracted from the images and the images label are recorded
parser.add_argument('--CAE_trained_gen_model_path',type=str,default='/code/trained_models/CAE_brain_trained_model.pt')
parser.add_argument('--style_dim',type=int,default=8)
parser.add_argument('--train_is',type=str,default='False')


opts = parser.parse_args()

if opts.cuda=='True':
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

###generator model loading
gen_model_path = opts.CAE_trained_gen_model_path
train_is=opts.train_is
style_dim=opts.style_dim
trainer = trainer(device=device,style_dim=style_dim,optim_para=None,gen_loss_weight_para=None,dis_loss_weight_para=None,train_is=train_is)
trainer.to(device)
state_dict_gen = torch.load(gen_model_path, map_location=device)
trainer.gen.load_state_dict(state_dict_gen['ab'])
trainer.eval()
encode = trainer.gen.encode
decode = trainer.gen.decode

####data preprocessing
transform_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
transform_list = [transforms.CenterCrop((256, 256))] + transform_list
transform_list = [transforms.Resize(256)] + transform_list
transform = transforms.Compose(transform_list)

####image label get
AB_image_name_label_path=opts.AB_image_name_label_path
img_name_label_dict={}
with open(AB_image_name_label_path)as file_name_label:
    lines_name_label=file_name_label.readlines()
for k in range(len(lines_name_label)):
    img_name_label_dict.update({lines_name_label[k].strip().split()[0]:lines_name_label[k].strip().split()[1]})

#####extract latents(class-associated codes)
A_img_path=opts.A_img_path
B_img_path=opts.B_img_path
latentAB_save_path=opts.latentAB_save_path

with open(latentAB_save_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # columns_name
    columns_name_list=['image_name']
    for latent_ids in range(style_dim):
        columns_name_list.append(str(latent_ids))
    columns_name_list.append('label')
    writer.writerow(columns_name_list)

    with torch.no_grad():
        ###get class-associated codes from A images
        img_n = 0
        for image_name in os.listdir(A_img_path):
            img_n=img_n+1
            example_path = A_img_path + image_name
            example_img=Variable(transform(Image.open(example_path).convert('RGB')).unsqueeze(0).to(
                device))
            c_A, s = encode(example_img)

            row_list = [image_name]
            s_show_str=''
            for j in range(s.size(1)):
                row_list.append(str(s[0][j].item()))
                s_show_str=s_show_str+str(s[0][j].item())+'  '

            row_list.append(img_name_label_dict[image_name])
            writer.writerow(row_list)

            print(str(img_n)+'  '+image_name+' CL codes: '+s_show_str)

        ###get class-associated codes from B images
        for image_name in os.listdir(B_img_path):
            img_n=img_n+1
            example_path = B_img_path + image_name
            example_img=Variable(transform(Image.open(example_path).convert('RGB')).unsqueeze(0).to(
                device))
            c_A, s = encode(example_img)

            row_list = [image_name]
            s_show_str=''
            for j in range(s.size(1)):
                row_list.append(str(s[0][j].item()))
                s_show_str=s_show_str+str(s[0][j].item())+'  '

            row_list.append(img_name_label_dict[image_name])
            writer.writerow(row_list)

            print(str(img_n)+'  '+image_name+' CL codes: '+s_show_str)

print("Latents(class-associated codes) extraction finished")


