import torch.utils.data as data
from PIL import Image
import os
import os.path
import torch



def default_loader(path):
    return Image.open(path).convert('RGB')

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = root+'/'+fname
                images.append(path)
    return images


class ImageFolder(data.Dataset):

    def __init__(self, root, label_path,transform=None, return_paths=False,
                 loader=default_loader):
        imgs = sorted(make_dataset(root))
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

        with open(label_path)as file:
            lines=file.readlines()
        self.label_all={}
        for i in range(len(lines)):
            self.label_all.update({lines[i].strip().split()[0]:lines[i].strip().split()[1]})



    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        label=self.label_all[path[(path.rfind('/')+1):]]
        label = torch.Tensor([int(label)])
        label = label.long()

        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img,label,path
        else:
            return img,label

    def __len__(self):
        return len(self.imgs)


