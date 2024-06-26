from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse, audio
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch, face_detection
from models import Wav2Lip
import platform
import numpy as np
import requests
import base64
import argparse
import cv2
from tqdm import tqdm
import cv2
from batch_face import RetinaFace
from PIL import Image
import pickle

# 创建命令行参数解析器，并设置描述
parser = argparse.ArgumentParser(description='利用 Wav2Lip 模型将视频中的嘴唇与音频同步的推理代码')

# 添加命令行参数，用于指定模型权重的检查点文件
parser.add_argument('--checkpoint_path', type=str,
					help='用于加载模型权重的检查点文件的名称', required=True)

# 添加命令行参数，用于指定包含人脸的视频或图片文件路径
parser.add_argument('--face', type=str,
					help='包含人脸使用的视频/图像文件路径', required=True)

# 添加命令行参数，用于指定作为原始音频源的视频或音频文件路径
parser.add_argument('--audio', type=str,
					help='用作原始音频源的视频/音频文件路径', required=True)

# 添加命令行参数，用于指定结果视频的保存路径，默认保存在 results/result_voice.mp4
parser.add_argument('--outfile', type=str, help='保存结果视频的路径。见默认示例。',
								default='results/result_voice.mp4')
parser.add_argument('--tmp', type=str, help='推理结果视频的临时路径。见默认示例。',
					default='temp/result.avi')

# 添加命令行参数，用于决定是否使用静态图像进行推理，默认为 False
parser.add_argument('--static', type=bool,
					help='如果为 True，则只使用第一帧视频进行推理', default=False)

# 添加命令行参数，用于指定帧率，默认为 25fps，仅当输入为静态图像时可指定
parser.add_argument('--fps', type=float, help='可以指定的帧率，默认为 25',
					default=25., required=False)

# 添加命令行参数，用于设置视频的填充区域，默认为上10，其它为0
parser.add_argument('--pads', nargs='+', type=int, default=[0, 20, 0, 0],
					help='填充（上，下，左，右）。请调整至少包含下巴')

# 添加命令行参数，用于指定面部检测的批量大小，默认为16
parser.add_argument('--face_det_batch_size', type=int,
					help='面部检测的批量大小', default=16)

# 添加命令行参数，用于指定 Wav2Lip 模型的批量大小，默认为128
parser.add_argument('--wav2lip_batch_size', type=int, help='Wav2Lip 模型的批量大小', default=128)

# 添加命令行参数，用于设置视频分辨率缩放因子，默认为1
parser.add_argument('--resize_factor', default=1, type=int,
			help='通过此因子减少分辨率。有时在 480p 或 720p 获得最佳效果')

# 添加命令行参数，用于设置视频裁剪区域，默认不裁剪
parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1],
					help='裁剪视频到较小区域（上，下，左，右）。在 resize_factor 和 rotate 参数之后应用。'
					'如果存在多个面部很有用。-1 表示根据高度、宽度自动推断值')

# 添加命令行参数，用于指定固定的面部边界框，默认为不指定
parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1],
					help='指定一个固定的面部边界框。仅作为最后手段使用，如果检测不到面部。'
					'同时，如果面部移动不多，可能会起作用。语法：（上，下，左，右）。')

# 添加命令行参数，用于设置是否旋转视频，默认为不旋转
parser.add_argument('--rotate', default=False, action='store_true',
					help='有时手机拍摄的视频可能会旋转 90 度。如果为真，将视频向右旋转 90 度。'
					'如果输入看起来正常的视频却得到翻转的结果时使用')

# 添加命令行参数，用于设置是否平滑面部检测，默认为进行平滑
parser.add_argument('--nosmooth', default=False, action='store_true',
					help='阻止在短时间窗口内平滑面部检测')


args = parser.parse_args()
args.img_size = 96
kernel = last_mask = x = y = w = h = None


if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
	args.static = True

#对检测到的人脸框位置进行平滑处理，以减少检测结果的抖动或突变。
#平滑算法
def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T :]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes

def exponential_smoothing(boxes, alpha):
    smoothed_boxes = []
    for i in range(len(boxes)):
        if i == 0:
            smoothed_boxes.append(boxes[i])
        else:
            smoothed_box = alpha * boxes[i] + (1 - alpha) * smoothed_boxes[i-1]
            smoothed_boxes.append(smoothed_box)
    return smoothed_boxes

# 打开模型文件
with open(os.path.join("checkpoints", "predictor.pkl"), "rb") as f:
    predictor = pickle.load(f)
	
with open(os.path.join("checkpoints", "mouth_detector.pkl"), "rb") as f:
    mouth_detector = pickle.load(f)

def create_mask(img, original_img):
    global kernel, last_mask, x, y, w, h # Add last_mask to global variables

    # Convert color space from BGR to RGB if necessary
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB, original_img)

    if last_mask is not None:
        last_mask = np.array(last_mask)  # Convert PIL Image to numpy array
        last_mask = cv2.resize(last_mask, (img.shape[1], img.shape[0]))
        mask = last_mask  # use the last successful mask
        mask = Image.fromarray(mask)

    else:
        # Detect face
        faces = mouth_detector(img)
        if len(faces) == 0:
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
            return img, None
        else:
            face = faces[0]
            shape = predictor(img, face)

            # Get points for mouth
            mouth_points = np.array(
                [[shape.part(i).x, shape.part(i).y] for i in range(48, 68)]
            )

            # Calculate bounding box dimensions
            x, y, w, h = cv2.boundingRect(mouth_points)

            # Set kernel size as a fraction of bounding box size
            kernel_size = int(max(w, h) * 1.5)
            # if kernel_size % 2 == 0:  # Ensure kernel size is odd
            # kernel_size += 1

            # Create kernel
            kernel = np.ones((kernel_size, kernel_size), np.uint8)

            # Create binary mask for mouth
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(mask, mouth_points, 255)

            # Dilate the mask
            dilated_mask = cv2.dilate(mask, kernel)

            # Calculate distance transform of dilated mask
            dist_transform = cv2.distanceTransform(dilated_mask, cv2.DIST_L2, 5)

            # Normalize distance transform
            cv2.normalize(dist_transform, dist_transform, 0, 255, cv2.NORM_MINMAX)

            # Convert normalized distance transform to binary mask and convert it to uint8
            _, masked_diff = cv2.threshold(dist_transform, 50, 255, cv2.THRESH_BINARY)
            masked_diff = masked_diff.astype(np.uint8)

            
            blur = 1
            # Set blur size as a fraction of bounding box size
            blur = int(max(w, h) * blur)  # 10% of bounding box size
            if blur % 2 == 0:  # Ensure blur size is odd
                blur += 1
            masked_diff = cv2.GaussianBlur(masked_diff, (blur, blur), 0)

            # Convert mask to single channel where pixel values are from the alpha channel of the current mask
            mask = Image.fromarray(masked_diff)

            last_mask = mask  # Update last_mask with the final mask after dilation and feathering

    # Convert numpy arrays to PIL Images
    input1 = Image.fromarray(img)
    input2 = Image.fromarray(original_img)

    # Resize mask to match image size
    # mask = Image.fromarray(mask)
    mask = mask.resize(input1.size)

    # Ensure images are the same size
    assert input1.size == input2.size == mask.size

    # Paste input1 onto input2 using the mask
    input2.paste(input1, (0, 0), mask)

    # Convert the final PIL Image back to a numpy array
    input2 = np.array(input2)

    # input2 = cv2.cvtColor(input2, cv2.COLOR_BGR2RGB)
    cv2.cvtColor(input2, cv2.COLOR_BGR2RGB, input2)

    return input2, mask


# 人脸检测
# def face_detect(images):
# 	#初始化人脸检测器
# 	#face_detection.FaceAlignment用于检测和对其人脸
# 	#face_detection.LandmarksType._2D 指定了检测到的人脸
# 	detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
# 											flip_input=False, device=device)

# 	batch_size = args.face_det_batch_size#设置模型批次
	
# 	while 1:
# 		predictions = []
# 		try:
# 			for i in tqdm(range(0, len(images), batch_size)):
# 				#detector.get_detections_for_betch() 进行检测人脸
# 				predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
# 		except RuntimeError: #内存溢出错误
# 			if batch_size == 1:
# 				raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
# 			batch_size //= 2 #将批次减半
# 			print('Recovering from OOM error; New batch size: {}'.format(batch_size))
# 			continue
# 		break

# 	results = []
# 	pady1, pady2, padx1, padx2 = args.pads
# 	for rect, image in zip(predictions, images):
# 		if rect is None:
# 			cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
# 			raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

# 		y1 = max(0, rect[1] - pady1)
# 		y2 = min(image.shape[0], rect[3] + pady2)
# 		x1 = max(0, rect[0] - padx1)
# 		x2 = min(image.shape[1], rect[2] + padx2)

		
# 		results.append([x1, y1, x2, y2])

# 	boxes = np.array(results)
# 	# if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5) #简单平滑
# 	# print(boxes)
# 	alpha = 0.2  # 指数加权移动平均的参数
# 	# 使用指数加权移动平均对人脸框位置进行平滑处理
# 	if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
# 	print('正在进行指数平均')
# 	# print(boxes)
# 	# 生成一个包含图像片段及其边界框坐标
# 	results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

# 	del detector
# 	return results 

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
gpu_id = 0 if torch.cuda.is_available() else -1

detector = RetinaFace(
        gpu_id=gpu_id, model_path="checkpoints/mobilenet.pth", network="mobilenet"
    ) # 加载RetinaFace
detector_model = detector.model


# 人脸检测
def face_detect(images):
    batch_size = args.face_det_batch_size  # 设置模型批次

    results = []
    pady1, pady2, padx1, padx2 = args.pads
    for i in tqdm(range(0, len(images), batch_size)):
        batch = images[i:i + batch_size]
        all_faces = detector(batch)  # 图像中所检测到的人脸列表

        for faces, image in zip(all_faces, batch):
            if faces:
                box, landmarks, score = faces[0]
                # 将坐标值转换为整数
                y1 = max(0, int(box[1] - pady1))
                y2 = min(image.shape[0], int(box[3] + pady2))
                x1 = max(0, int(box[0] - padx1))
                x2 = min(image.shape[1], int(box[2] + padx2))

                results.append([x1, y1, x2, y2])
            else:
                cv2.imwrite('temp/faulty_frame.jpg', image)  # check this frame where the face was not detected.
                raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

    boxes = np.array(results)
    alpha = 0.2  # 指数加权移动平均的参数
    # 使用指数加权移动平均对人脸框位置进行平滑处理
    if not args.nosmooth:
        boxes = get_smoothened_boxes(boxes, T=5)
    print('正在进行简单平滑')

    # 生成一个包含图像片段及其边界框坐标
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    return results




# 生成器函数，用于生成训练数据的批次
def datagen(frames, mels):
	img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	# 根据args.box参数来选择是否使用人脸检测
	if args.box[0] == -1:
		if not args.static:
			face_det_results = face_detect(frames) # BGR2RGB for CNN face detection
		else:
			face_det_results = face_detect([frames[0]])
	else:
		print('Using the specified bounding box instead of face detection...')
		y1, y2, x1, x2 = args.box
		face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

	for i, m in enumerate(mels):
		idx = 0 if args.static else i%len(frames)
		frame_to_save = frames[idx].copy()
		face, coords = face_det_results[idx].copy()

		face = cv2.resize(face, (args.img_size, args.img_size))
			
		img_batch.append(face)
		mel_batch.append(m)
		frame_batch.append(frame_to_save)
		coords_batch.append(coords)

		if len(img_batch) >= args.wav2lip_batch_size:
			img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

			img_masked = img_batch.copy()
			img_masked[:, args.img_size//2:] = 0

			img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
			mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

			yield img_batch, mel_batch, frame_batch, coords_batch
			img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if len(img_batch) > 0:
		img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

		img_masked = img_batch.copy()
		img_masked[:, args.img_size//2:] = 0

		img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
		mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

		yield img_batch, mel_batch, frame_batch, coords_batch

mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

def _load(checkpoint_path):
	if device == 'cuda':
		checkpoint = torch.load(checkpoint_path)
	else:
		checkpoint = torch.load(checkpoint_path,
								map_location=lambda storage, loc: storage)
	return checkpoint

# 加载wav2lip模型
def load_model(path):
	model = Wav2Lip()
	print("Load checkpoint from: {}".format(path))
	checkpoint = _load(path)
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)

	model = model.to(device)
	return model.eval()



# 用于处理输入视频或图像文件并准备进行推理操作
def main():
	
	if not os.path.isfile(args.face):
		raise ValueError('--face argument must be a valid path to video/image file')

	elif args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
		full_frames = [cv2.imread(args.face)] # 加载图片
		fps = args.fps

	else:
		video_stream = cv2.VideoCapture(args.face) # 读取视频帧
		fps = video_stream.get(cv2.CAP_PROP_FPS)

		print('Reading video frames...')

		full_frames = []
		while 1:
			still_reading, frame = video_stream.read()
			if not still_reading:
				video_stream.release()
				break
			if args.resize_factor > 1:
				frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor)) #缩放

			if args.rotate:
				frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE) # 旋转

			y1, y2, x1, x2 = args.crop # 裁减
			if x2 == -1: x2 = frame.shape[1]
			if y2 == -1: y2 = frame.shape[0]

			frame = frame[y1:y2, x1:x2]

			full_frames.append(frame) # 保存视频帧

	print ("Number of frames available for inference: "+str(len(full_frames)))

	# 音频处理
	if not args.audio.endswith('.wav'):
		print('Extracting raw audio...')
		command = 'ffmpeg -y -i {} -strict -2 {}'.format(args.audio, 'temp/temp.wav') # 音频文件格式转换

		subprocess.call(command, shell=True)
		args.audio = 'temp/temp.wav'

	wav = audio.load_wav(args.audio, 16000)
	mel = audio.melspectrogram(wav)
	print(mel.shape)

	if np.isnan(mel.reshape(-1)).sum() > 0:
		raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

	mel_chunks = []
	mel_idx_multiplier = 80./fps  # 帧移 
	i = 0
	while 1:
		start_idx = int(i * mel_idx_multiplier)
		if start_idx + mel_step_size > len(mel[0]):
			mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
			break
		mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
		i += 1

	print("Length of mel chunks: {}".format(len(mel_chunks)))

	full_frames = full_frames[:len(mel_chunks)]

	batch_size = args.wav2lip_batch_size
	gen = datagen(full_frames.copy(), mel_chunks) # 生成特征对

	for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, 
											total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
		if i == 0:
			model = load_model(args.checkpoint_path)
			print ("Model loaded")

			frame_h, frame_w = full_frames[0].shape[:-1]
			out = cv2.VideoWriter(args.tmp, 
									cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h)) #写入临时视频
			print(frame_w,frame_h)
		img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
		mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

		with torch.no_grad():
			pred = model(mel_batch, img_batch) # 预测
		pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.


		# 视频处理过程
		# for p, f, c in tqdm(zip(pred, frames, coords), total=len(pred), desc='Processing frames'):
		# 	y1, y2, x1, x2 = c
			
		# 	# 缩放推理后的唇形为原视频人脸大小
		# 	p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
		# 	f[y1:y2, x1:x2] = p
		# 	# 对当前帧进行人像增强处理
		# 	# 将处理后的帧写入输出视频
		# 	out.write(f)
		# 	# out.write(f)
		for p, f, c in zip(pred, frames, coords):
		# cv2.imwrite('temp/f.jpg', f)

			y1, y2, x1, x2 = c


			p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
			cf = f[y1:y2, x1:x2]
			
			p, last_mask = create_mask(p, cf)

			f[y1:y2, x1:x2] = p

			out.write(f)
  

	out.release()

	command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio, args.tmp, args.outfile)
	subprocess.call(command, shell=platform.system() != 'Windows')

if __name__ == '__main__':
	main()
