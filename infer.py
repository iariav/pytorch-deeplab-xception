import os
from modeling.deeplab import *
from dataloaders.datasets import arbel
import glob
import time
import torch.backends.cudnn as cudnn

from IPython import display
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms

## Load model in Pytorch

_CHECKPOINT_PATH = '/home/ido/Deep/Pytorch/pytorch-deeplab-xception/run/arbel/deeplab-xception/experiment_8/checkpoint_fp16_02_os8.pth'
_PARAMS_PATH = '/home/ido/Deep/Pytorch/pytorch-deeplab-xception/run/arbel/deeplab-xception/experiment_8/parameters.txt'
INPUT_SIZE_HEIGHT = 720
INPUT_SIZE_WIDTH = 1280 # was 1280

cudnn.benchmark = True
cudnn.enabled = True

file = open(_PARAMS_PATH, 'r')
parameters = {}
for line in file:

   k, v = line.split(':')
   parameters[k] = v.rstrip()

# Define network
model = DeepLab(num_classes=arbel.ArbelSegmentation.NUM_CLASSES,
                backbone=parameters['backbone'],
                output_stride=int(parameters['out_stride']),
                sync_bn=True,
                freeze_bn=False)

if not os.path.isfile(_CHECKPOINT_PATH):
    raise RuntimeError("=> no checkpoint found at '{}'".format(_CHECKPOINT_PATH))
checkpoint = torch.load(_CHECKPOINT_PATH)

model.load_state_dict(checkpoint['state_dict'])
model = model.eval()
model = model.cuda()
torch.set_grad_enabled(False)

print("=> loaded checkpoint '{}')".format(_CHECKPOINT_PATH))


LABEL_NAMES = np.asarray([
    'Terrain','Unpaved route','Terrain route - Metal palette','Tree - trunk','Tree - leaf','Rocks','Large Shrubs',
    'Low Vegetation','Wire Fance','Background(sky)','Person','Vehicle','building','Paved Road','Misc'
    , 'Ignore', 'Ignore', 'Ignore', 'Ignore'
])

labels = []

colormap = np.array(
           [[149,129,107], # Terrain
           [198,186,173], # Terrain route
           [77,69,59],    # Terrain route - Metal palette
           [144,100,0],   # Tree - trunk
           [103,128,88],  # Tree - leaf
           [128,0,255], # Rocks
           [158,217,92],  # Large Shrubs
           [200,217,92],  # Low Vegetation
           [217,158,93],  # Wire Fance
           [0,0,128],     # background(sky)
           [160,5,5],     # Person
           [53,133,193],  # Vehicle
           [0,0,0],       # building
           [30,30,30],    # paved road
           [255,255,255], # Misc.
           [0, 255, 255],  # empty
           [0, 255, 255],  # empty
           [0, 255, 255],  # empty
           [0, 255, 255],  # empty
           ],   
           dtype=np.float32) 

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = colormap[FULL_LABEL_MAP]


def list_images(folder, pattern='*', ext='bmp'):
    """List the images in a specified folder by pattern and extension

    Args:
        folder (str): folder containing the images to list
        pattern (str, optional): a bash-like pattern of the files to select
                                 defaults to * (everything)
        ext(str, optional): the image extension (defaults to png)

    Returns:
        str list: list of (filenames) images matching the pattern in the folder
    """
    filenames = sorted(glob.glob(folder + pattern + '.' + ext))
    return filenames


def save_segmentation(image_name, seg_map):
    seg_image = colormap[seg_map.squeeze()].astype(np.uint8)
    segmentation = Image.fromarray(seg_image)
    segmentation = segmentation.resize(size=(1280, 960), resample=Image.NEAREST)
    newfilename = os.path.join(os.path.dirname(image_name),
                               os.path.splitext(os.path.basename(image_name))[0] + '_ID_PYTORCH.png')
    segmentation.save(newfilename, "PNG")

    # seg_image_crf = colormap[seg_map_crf].astype(np.uint8)
    # segmentation_crf = Image.fromarray(seg_image_crf)
    # segmentation_crf = segmentation_crf.resize(size=(1280, 960), resample=Image.NEAREST)
    # newfilename_crf = os.path.join(os.path.dirname(image_name),
    #                            os.path.splitext(os.path.basename(image_name))[0] + '_ID_CRF.png')
    # segmentation_crf.save(newfilename_crf, "PNG")

## Run on sample images

IMAGE_DIR = '/home/ido/Deep/SegmentationDatasets/elyakim/drive3/'
images = list_images(IMAGE_DIR, ext='jpg')

trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])

target_size = (int(INPUT_SIZE_WIDTH), int(INPUT_SIZE_HEIGHT))

for image_name in images:
    image_path = os.path.join(IMAGE_DIR, image_name)

    image = Image.open(image_path).convert('RGB').resize(target_size, Image.ANTIALIAS)
    print('running deeplab on image %s...' % image_name)
    start_tot = time.time()
    image_t = trans(image).cuda()
    with torch.no_grad():
        pred = model(image_t.unsqueeze(0)).squeeze().cpu().numpy()
    pred = np.argmax(pred, axis=0)

    # save_segmentation(image_name, pred)

    end_tot = time.time()
    print("Classification took {} ms".format((end_tot - start_tot) * 1000))





