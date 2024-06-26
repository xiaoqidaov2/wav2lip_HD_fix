from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
import os
import subprocess
import tempfile
import shutil
import time

app = FastAPI()

LOCK_FILE = "command_lock.lock"

model_list = {'视频生成数字人模型': 'checkpoints/wav2lip_gan.pth'}

def save_file(file_obj):
    tmpdir = f"/home/featurize/work/ai_addons/wav2lip/Wav2Lip/temp/{t}"
    print('临时文件夹地址：{}'.format(tmpdir))
    print('上传文件的地址：{}'.format(file_obj.filename))  # 输出上传后的文件在gradio中保存的绝对地址
    # 获取到上传后的文件的绝对路径后，其余的操作就和平常一致了
    if os.path.isdir(tmpdir):
        # print('文件夹已存在')
        pass
    else:
        os.makedirs(tmpdir)
    # 将文件内容保存到临时目录中
    file_path = os.path.join(tmpdir, file_obj.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file_obj.file, buffer)
    print('上传的文件已保存到：{}'.format(file_path))
    return file_path


def acquire_lock():
    while os.path.isfile(LOCK_FILE):
        time.sleep(0.1)  # 等待锁文件释放
    with open(LOCK_FILE, "w") as f:
        f.write("Locked")


def release_lock():
    if os.path.isfile(LOCK_FILE):
        os.remove(LOCK_FILE)


def simple_demo_greet(checkpoint, video_file, audio_file):
    global t
    t = str(time.time()).split(".")
    t = t[0] + t[1]
    face_path = save_file(video_file)
    audio_path = save_file(audio_file)
    print(model_list[checkpoint], face_path, audio_path)

    out_file = f'results/{t}.mp4'
    tmp_file = f'temp/{t}.avi'
    print(out_file, tmp_file)

    acquire_lock()  # 获取锁

    # 构建命令
    command = f'python3.7 inference.py --checkpoint_path {model_list[checkpoint]} --face {face_path} --audio {audio_path} --outfile {out_file} --tmp {tmp_file}'
    print(command)
    # 启动一个进程并捕获其输出
    process = subprocess.Popen(command, shell=True)
    process.wait()  # 等待命令执行完成

    release_lock()  # 释放锁

    return out_file


def senior_demo_greet(checkpoint, video_file, audio_file, retouching):
    global t
    t = str(time.time()).split(".")
    t = t[0] + t[1]
    face_path = save_file(video_file)
    audio_path = save_file(audio_file)
    print(model_list[checkpoint], face_path, audio_path)

    out_file = f'results/{t}.mp4'
    tmp_file = f'temp/{t}.avi'
    print(out_file, tmp_file)

    acquire_lock()  # 获取锁

    # 构建命令
    command = f'python3.7 inference.py --checkpoint_path {model_list[checkpoint]} --face {face_path} --audio {audio_path} --outfile {out_file} --tmp {tmp_file}'

    # 启动一个进程并捕获其输出
    process = subprocess.run(command, shell=True)
    print('结束唇形合成')
    if retouching:
        print('启动人脸修复')
        command = f'python gg.py --video {out_file} --audio {audio_path} --outfile {out_file}'
        process = subprocess.run(command, shell=True)

    release_lock()  # 释放锁

    return out_file

# 定义 FastAPI 路由
@app.post("/simple_demo")
async def simple_demo_api(checkpoint: str = Form(...), video_file: UploadFile = File(...), audio_file: UploadFile = File(...)):
    output_file = simple_demo_greet(checkpoint, video_file, audio_file)
    return FileResponse(output_file, media_type='video/mp4')

@app.post("/senior_demo")
async def senior_demo_api(checkpoint: str = Form(...), video_file: UploadFile = File(...), audio_file: UploadFile = File(...), retouching: bool = False):
    output_file = senior_demo_greet(checkpoint, video_file, audio_file, retouching)
    return FileResponse(output_file, media_type='video/mp4')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
