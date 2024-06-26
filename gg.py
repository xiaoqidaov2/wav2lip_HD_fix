import cv2
from tqdm import tqdm
import numpy as np
import warnings
import os
import requests
import base64
import cv2
import torch
from torchvision import models, transforms
from PIL import Image
from gfpgan.utils import GFPGANer
from realesrgan.utils import RealESRGANer
from basicsr.archs.srvgg_arch import SRVGGNetCompact
from IPython.display import display
import os
import requests
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline 
from torchvision.transforms import ToTensor, Normalize, ConvertImageDtype
import subprocess
import platform
import argparse
import time
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser(description='人像修复')

# 添加命令行参数，用于指定包含人脸的视频或图片文件路径
parser.add_argument('--video', type=str,
					help='包含人脸使用的视频/图像文件路径', required=True)

# 添加命令行参数，用于指定作为原始音频源的视频或音频文件路径
parser.add_argument('--audio', type=str,
					help='用作原始音频源的视频/音频文件路径', required=True)

# 添加命令行参数，用于指定结果视频的保存路径，默认保存在 results/result_voice.mp4
parser.add_argument('--outfile', type=str, help='保存结果视频的路径。见默认示例。',
								default='results/result_voice.mp4')

args = parser.parse_args()
# Load RealESRGAN mod
realesrgan_model_path = 'weights/realesr-general-x4v3.pth'

# Initialize RealESRGAN
sr_model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
half = True if torch.cuda.is_available() else False
realesrganer = RealESRGANer(scale=1, model_path=realesrgan_model_path, model=sr_model, tile=0, tile_pad=10, pre_pad=0, half=half)




# Load GFPGAN model
gfpgan_model_path = 'weights/GFPGANv1.4.pth'

# Initialize GFPGAN
face_enhancer = GFPGANer(model_path=gfpgan_model_path, upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=realesrganer)



def process_video(video_path, output_path):
    print('开始人脸修复')
    # 打开视频文件
    video = cv2.VideoCapture(video_path)
    
    # 获取视频的帧率和总帧数
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 获取视频的宽度和高度
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 创建输出视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print(width, height)
    number = 0
    # 遍历视频的每一帧
    for _ in tqdm(range(total_frames), desc="Processing frames"):
        ret, frame = video.read()
        if not ret:
            break
        
        # 对当前帧进行人像增强处理
        _, _, result = face_enhancer.enhance(frame, has_aligned=False, only_center_face=False, paste_back=True)
        print(result.shape)
        # print(result)
        output_video.write(result)
    
    # 释放资源
    video.release()
    output_video.release()

t = str(time.time()).split(".")
t = t[0] + t[1]
# 指定输入视频路径和输出视频路径
input_video = args.video
input_audio = args.audio
tmp_video = tmp_file = f'temp/{t}.avi'
output_video = args.outfile
print(tmp_video)
if __name__ == '__main__':
    # 调用函数处理视频
    process_video(input_video, tmp_video)
    command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(input_audio, tmp_video,output_video)
    subprocess.call(command, shell=platform.system() != 'Windows')