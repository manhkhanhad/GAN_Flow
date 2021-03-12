import os
from torchvision import transforms
from PIL import Image
import torch

class PairDataset():
    def __init__(self,opt):
        # tu AB_path -> tensor A,B -> dictionary data{'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}  (Unaligned_dataset.py)
        # dictionary data -> pytorch DataLoader     (data/__init__.py class CustomDatasetDataLoader() )
        # Ham model.set_input : tao self.real_A vaf self.real_B de dua vao train (pix2pix.py def set_input)


        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        if opt.direction == 'AtoB':
            input_nc = opt.input_nc
            output_nc = opt.output_nc
        else:
            input_nc = opt.output_nc
            output_nc = opt.input_nc

        trainA_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
        trainB_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])


    def __getitem__(self, index):
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        A = self.transform_A(A_img)
        A = self.transform_A(A_img)
        return {'A': A, 'B': B, 'A_paths': self.A_paths, 'B_paths': self.B_paths}

    def __len__(self):
        return len(self.A_paths)


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, f_names in sorted(os.walk(dir)):
        for f_name in f_names:
            if is_image_file(f_name):
                path = os.path.join(root, f_name)
                images.append(path)
    return images

def create_dataset(opt):

    data_loader = PairDataset(opt)
    dataloader = torch.utils.data.DataLoader(
                                self.dataset,
                                batch_size=opt.batch_size,
                                shuffle= opt.shuffle,
                                num_workers=int(opt.num_threads))
    return dataloader
