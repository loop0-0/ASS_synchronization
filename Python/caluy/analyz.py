import multiprocessing
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import sys
import functools
import os
import os.path
import threading
from timeit import default_timer as timer

import cv2
import numpy as np
#---------------------------------------
class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

class RRDBNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=4):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.sf = sf

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        if self.sf==4:
            self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        if self.sf==4:
            fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out
#---------------------------------------



image_path          = "test.jpg"
AI_model            = "BSRGAN" 
device              = "cpu"
upscale_factor      = 2
cut_image_factor    = 170
actual_step         = ""
single_file         = True
multiple_files      = False
video_files         = False
multi_img_list             = []
video_frames_list          = []
video_frames_upscaled_list = []
original_video_path = ""
#----------------------------------------
def imsave(img, img_path):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)

def imwrite(img, img_path):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)

def uint2tensor4(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.).unsqueeze(0)

def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0).round())


def images_to_check_convert_filenames(image_list, AI_model, upscale_factor):
    temp_images_to_delete = []
    image_list_to_check = []
    
    for image in image_list:
        temp = image.replace(".png","") + "_" + AI_model + "_x" + str(upscale_factor) + ".png"
        image_list_to_check.append(temp)
        temp_images_to_delete.append(image.replace(".png","_resized.png"))

    return image_list_to_check, temp_images_to_delete
def thread_wait_for_multiple_file(image_list, AI_model, upscale_factor):
    start     = timer()
    image_list_to_check, temp_images_to_delete = images_to_check_convert_filenames(image_list,
                                                                                     AI_model, 
                                                                                     upscale_factor)
    # check if files exist
    how_many_images = len(image_list_to_check)
    counter_done    = 0
    for image in image_list_to_check:
        while not os.path.exists(image):
            time.sleep(1)

        if os.path.isfile(image):
            counter_done += 1
            info_string.set("Upscaled images " + str(counter_done) + "/" + str(how_many_images))

        if counter_done == how_many_images:  
            # delete temp files
            if len(temp_images_to_delete) > 0:
                for to_delete in temp_images_to_delete:
                    if os.path.exists(to_delete):
                        os.remove(to_delete)  

            end       = timer()
            info_string.set("Upscale completed [" + str(round(end - start)) + " sec.]")
            place_upscale_button()


def optimize_torch(device):
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)
    if 'cpu' in device:
        torch.set_num_threads(4)
    elif 'cuda' in device:
        torch.backends.cudnn.benchmark = True

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

            
def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


def find_file_by_relative_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(
        os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

def prepare_torch_model(AI_model, device):
    if 'cpu' in device:
        backend = torch.device('cpu')
    elif 'cuda' in device:
        backend = torch.device('cuda')
    
    model_path = find_file_by_relative_path(AI_model + ".pth")
    
    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=4)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()

    for _ , v in model.named_parameters():
        v.requires_grad = False
    
    # not working, maibe torch > 1.8.2 ?
    # model = torch.jit.optimize_for_inference(torch.jit.script(model.eval()))
    
    return model.to(backend, non_blocking=True)


def torch_AI_upscale_multiple_images(image_list, AI_model, upscale_factor, device):
    try:
        # 0) define the model
        model = prepare_torch_model(AI_model, device)
        optimize_torch(device)

        # 1) resize all images
        downscaled_images = []
        for image in image_list:
            img_downscaled = resize_image(image, upscale_factor)
            downscaled_images.append(img_downscaled)

        
        for img in downscaled_images:
            result_path = (img.replace("_resized.png","").replace(".png","") + 
                            "_"  + AI_model +
                            "_x" + str(upscale_factor) + 
                            ".png")

            # 2) calculating best slice number
            img_tmp = cv2.imread(img)
            val = min(img_tmp.shape[1], img_tmp.shape[0])
            num_tiles = round(val/cut_image_factor)
            if (num_tiles % 2) != 0:
                num_tiles += 1

            # 3) divide the image in tiles
            tiles = slice_image(img, num_tiles)

            # 4) upscale each tiles
            with torch.no_grad():
                for tile in tiles:
                    tile_adapted  = adapt_image_for_deeplearning(tile.filename)
                    tile_adapted  = tile_adapted.to(device, non_blocking = True)
                    tile_upscaled = model(tile_adapted)
                    tile_upscaled = tensor2uint(tile_upscaled)
                    imsave(tile_upscaled, tile.filename)
                    tile.image  = Image.open(tile.filename)
                    tile.coords = (tile.coords[0]*4, tile.coords[1]*4)

            # 5) then reconstruct the image by tiles upscaled
            image_upscaled = reunion_image(tiles)
        
            # 6) remove tiles file
            delete_tiles_from_disk(tiles)

            # 7) save reconstructed image
            cv2.imwrite(result_path, image_upscaled)
    except:
        error_root = tkinterDnD.Tk()
        ErrorMessage(error_root,  "upscale_problem")




    

process_upscale = multiprocessing.Process(target = torch_AI_upscale_multiple_images, 
                                            args   = (multi_img_list, AI_model, upscale_factor, device))
process_upscale.start()

thread_wait = threading.Thread(target = thread_wait_for_multiple_file, 
                                args   = (multi_img_list, AI_model, upscale_factor),
                                daemon = True)
thread_wait.start()
