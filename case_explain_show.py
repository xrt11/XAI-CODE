import os
import torch
from trainer_exchange import trainer
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
import torchvision.utils as vutils
from torch.nn import functional
import numpy as np
import cv2

# the path of the explained example and the reference counter example
example_path='example(explained)_DRUSEN-64262-2.jpeg'
counter_example_path='counter-example(reference)_NORMAL-8614915-1.jpeg'

# the path where generative results are saved
gen_img_save_path='CAE_explain_case_show/'

# distances between generative class-style codes and the one from the explained example along with the guided path formed by the example and the reference counter example
distances=[0,0.2,0.4,0.6,0.8,1]

# the path of the model which has been trained for synthesizing new samples
input_gen_model_path='CAE_gen_model.pt'

# the path of the outer classifier whose predictions would be explained using our method
outer_classifier_path='mobilenet_v2(classifier).pth'

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# function of saliency map generation based the difference map
def heatmap_show(image,difference_map):
    image = image.data.cpu().numpy()
    saliency_map = difference_map.data.cpu().numpy()
    saliency_map = saliency_map - saliency_map.min()
    saliency_map = saliency_map / saliency_map.max()
    saliency_map = saliency_map.clip(0, 1)
    saliency_map = np.uint8(saliency_map * 255).transpose(1, 2, 0)
    image = np.uint8(image * 255).transpose(1, 2, 0)
    color_heatmap = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)

    # Combine image with heatmap
    img_with_heatmap = np.float32(color_heatmap) + np.float32(image)
    img_with_heatmap01 = img_with_heatmap / np.max(img_with_heatmap)
    img_with_heatmap=np.uint8(255 * img_with_heatmap01)
    return color_heatmap, img_with_heatmap

# load the model used for generating new samples (including the encoder and the decoder)
trainer = trainer(device=device)
state_dict_gen = torch.load(input_gen_model_path,map_location=device)
trainer.gen.load_state_dict(state_dict_gen['ab'])
trainer.to(device)
trainer.eval()
encode = trainer.gen.encode
decode = trainer.gen.decode

# data preprocessing
transform_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
transform_list = [transforms.CenterCrop((256, 256))] + transform_list
transform_list = [transforms.Resize(256)] + transform_list
transform = transforms.Compose(transform_list)

# create path that would be used for saving the results (including synthetic samples and the saliency map/heatmap used for explanation)
if not os.path.exists(gen_img_save_path):
    os.makedirs(gen_img_save_path)

# load the classifier whose predictions for the explained example would be explained
classify_model=torch.load(outer_classifier_path,map_location=device)
classify_model=classify_model.eval()
classify_model=classify_model.to(device)


with torch.no_grad():
    # load and preprocess the explained example and the reference counter example
    example_img=Variable(transform(Image.open(example_path).convert('RGB')).unsqueeze(0).to(
        device))
    counter_example_img = Variable(
        transform(Image.open(counter_example_path).convert('RGB')).unsqueeze(0).to(
            device))

    # encode for extracting the class-style and the individual-style codes of the example and the reference counter example, here c represent the individual/content codes while s indicate the class-style codes
    c_A, s_A = encode(example_img)
    c_B, s_B=encode(counter_example_img)

    # predict the class of the explained example using the outer classifier
    pre_img_a = classify_model(example_img)
    pre_img_a = functional.softmax(pre_img_a, dim=1)
    pre_img_a_out_max, pre_img_a_out_max_index = pre_img_a.max(dim=1)
    print('example image is predicted as: '+str(pre_img_a_out_max_index.item()))

    for kr in distances:
        # generate new codes with different differences from the example's along with the guided path
        s_B_gen = s_A+kr*(s_B-s_A)

        # combine the individual-style code(c_A) of the explained example with the generative class-style codes(s_B_gen) for synthesizing new samples(out_b)
        out_b = decode(c_A, s_B_gen)

        # save the synthetic samples
        vutils.save_image((out_b.data + 1) / 2,
                          gen_img_save_path + 'output_cA(example)_sB(' +str(kr)+'d)'+'.jpg',
                          padding=0, normalize=False)

        # predict the class of the synthetic samples using outer classifier
        pre_img_sbca = classify_model(out_b)
        pre_img_sbca = functional.softmax(pre_img_sbca, dim=1)
        pre_img_sbca_out_max, pre_img_sbca_out_max_index = pre_img_sbca.max(dim=1)
        print('generative image with '+ str(kr)+'d distance from the example is predicted as: ' + str(pre_img_sbca_out_max_index.item()))

        # with the guided path, when the predicted class of the synthetic sample start to change, counter example is obtained for saliency map generation
        if pre_img_sbca_out_max_index.item()!=pre_img_a_out_max_index.item():

            # compute the difference between the explained example and the generative counter example
            difference_casb_ca = torch.abs(((out_b + 1) / 2) - ((example_img + 1) / 2))

            # generate the saliency map/heatmap based the difference between the explained example and the generative counter example
            colormap, colormap_with_img = heatmap_show((example_img.squeeze(0) + 1) / 2, difference_casb_ca.squeeze(0))

            # save the saliency map
            cv2.imwrite(gen_img_save_path + 'saliency_map_with_img_' + str(kr) + '.jpg', colormap_with_img)
            print('When the distance between generative case and explained example reach '+str(kr)
                  +'d , prediction result changes and the final saliency map with image is outputed.'
                  +' Continuously changing generative examples with the final saliency map is presented in '+gen_img_save_path)
            break

