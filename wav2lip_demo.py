import gradio as gr
import os
import subprocess
import tempfile
import shutil
import time

LOCK_FILE = "command_lock.lock"

model_list = {'视频生成数字人模型-无优化' : 'checkpoints/wav2lip_gan.pth','视频生成数字人模型-停顿优化' : 'checkpoints/wav2lip.pth'}
def save_file(file_obj):
    tmpdir = f"/home/featurize/work/ai_addons/wav2lip/Wav2Lip/temp/{t}"
    print('临时文件夹地址：{}'.format(tmpdir))
    print('上传文件的地址：{}'.format(file_obj)) # 输出上传后的文件在gradio中保存的绝对地址
    #获取到上传后的文件的绝对路径后，其余的操作就和平常一致了
    if os.path.isdir(tmpdir):
        # print('文件夹已存在')
        pass
    else:
        os.makedirs(tmpdir)
    # 将文件复制到临时目录中
    shutil.copy(file_obj, tmpdir)

    # 获取上传Gradio的文件名称
    FileName=os.path.basename(file_obj)

    # 获取拷贝在临时目录的新的文件地址
    NewfilePath=os.path.join(tmpdir,FileName)
    print(NewfilePath)

    # 打开复制到新路径后的文件
    with open(NewfilePath, 'rb') as file_obj:

        #在本地电脑打开一个新的文件，并且将上传文件内容写入到新文件
        outputPath=os.path.join(tmpdir,"New"+FileName)
        with open(outputPath,'wb') as w:
            w.write(file_obj.read())

    # 返回新文件的的地址（注意这里）
    return outputPath

# 测试函数
# file = ...  # 上传的文件对象
# save_dir = "/home/featurize/work/ai_addons/wav2lip/Wav2Lip/temp"
# saved_file_path = save_file(file, save_dir)



def acquire_lock():
    while os.path.isfile(LOCK_FILE):
        time.sleep(0.1)  # 等待锁文件释放
    with open(LOCK_FILE, "w") as f:
        f.write("Locked")

def release_lock():
    if os.path.isfile(LOCK_FILE):
        os.remove(LOCK_FILE)

        
        
def simple_demo_greet(*params):
    print(params)
    # 确定保存文件的目录
    # 保存 face 文件和 audio 文件
    global t
    t = str(time.time()).split(".")
    t = t[0] + t[1]
    face_path = save_file(params[1])
    audio_path = save_file(params[2])
    print(model_list[params[0]],face_path,audio_path)
    
    out_file = f'results/{t}.mp4'
    tmp_file = f'temp/{t}.avi'
    print(out_file, tmp_file)
    
    acquire_lock()  # 获取锁
    
    
    # 构建命令
    command = f'python3 inference.py --checkpoint_path {model_list[params[0]]} --face {face_path} --audio {audio_path} --outfile {out_file} --tmp {tmp_file}'
    print(command)
    # 启动一个进程并捕获其输出
    process = subprocess.Popen(command, shell=True)
    process.wait()  # 等待命令执行完成
    
    command = f'python3 gg.py --video {out_file} --audio {audio_path} --outfile {out_file}'
    print(command)
    process = subprocess.Popen(command, shell=True)
    process.wait()  # 等待命令执行完成
    
    release_lock()  # 释放锁
    
    return out_file

def senior_demo_greet(*params):
    print(params)
    # 确定保存文件的目录
    # 保存 face 文件和 audio 文件
    global t
    t = str(time.time()).split(".")
    t = t[0] + t[1]
    face_path = save_file(params[1])
    audio_path = save_file(params[2])
    print(model_list[params[0]],face_path,audio_path)
    
    out_file = f'results/{t}.mp4'
    tmp_file = f'temp/{t}.avi'
    print(out_file, tmp_file)
    
    acquire_lock()  # 获取锁
    
    
    # 构建命令
    command = f'python3 inference.py --checkpoint_path {model_list[params[0]]} --face {face_path} --audio {audio_path} --outfile {out_file} --tmp {tmp_file}'
   
    # 启动一个进程并捕获其输出
    process = subprocess.run(command, shell=True)
    print('结束唇形合成')
    if params[3]:
        print('启动人脸修复')
        command = f'python3 gg.py --video {out_file} --audio {audio_path} --outfile {out_file}'
        process = subprocess.run(command, shell=True)

    release_lock()  # 释放锁
    
    return out_file

import gradio as gr

def function1(input1):
    return f"处理结果: {input1}"

def function2(input2):
    return f"分析结果: {input2}"

simple_demo = gr.Interface(
    fn=simple_demo_greet,
    title='数字人演示demo',
    description="<center><div><h2>方法一：上传视频和音频</h2><p>1. 请在 <strong>Checkpoint</strong> 中选择你想要的模型。</p><p>2. 点击 <strong>Face Video/Image Path</strong> 上传含有人脸的视频或图片。</p><p>3. 点击 <strong>Audio</strong> 上传人说话的音频，点击提交等待结果。</p></div><div><h2>方法二：使用测试用例</h2><p>点击examples下的一个例子，并点击提交等待结果</p></div></center>",
    examples=[["视频生成数字人模型-停顿优化","test.mp4",'yinping.wav']],
    inputs=[
        gr.inputs.Dropdown(label="Checkpoint Path ",choices=['视频生成数字人模型-无优化','视频生成数字人模型-停顿优化'],default='视频生成数字人模型-停顿优化'),  # 文本框输入
        gr.inputs.Video(label="Face Video/Image Path (.mp4, .avi)"),  # 文件上传
        gr.inputs.Audio(label="Audio(.mp3, .wav)",type = 'filepath'),  # 文件上传
    ],
    outputs=["video"],
)

senior_demo = gr.Interface(
    fn=senior_demo_greet,
    title='数字人演示demo',
    description="<center><div><h2>方法一：上传视频和音频</h2><p>1. 请在 <strong>Checkpoint</strong> 中选择你想要的模型。</p><p>2. 点击 <strong>Face Video/Image Path</strong> 上传含有人脸的视频或图片。</p><p>3. 点击 <strong>Audio</strong> 上传人说话的音频，点击提交等待结果。</p></div><div><h2>方法二：使用测试用例</h2><p>点击examples下的一个例子，并点击提交等待结果</p></div></center>",
    examples=[["视频生成数字人模型-停顿优化","test.mp4",'yinping.wav',True]],
    inputs=[
        gr.inputs.Dropdown(label="Checkpoint Path ",choices=['视频生成数字人模型-无优化','视频生成数字人模型-停顿优化'],default='视频生成数字人模型-停顿优化'),  # 文本框输入
        gr.inputs.Video(label="Face Video/Image Path (.mp4, .avi)"),  # 文件上传
        gr.inputs.Audio(label="Audio(.mp3, .wav)",type = 'filepath'),  # 文件上传
        gr.inputs.Checkbox(label='Face retouching', default=True), # 复选框
    ],
    outputs=["video"],
)

tabbed_interface = gr.TabbedInterface([simple_demo, senior_demo], ["简单演示", "高级演示"])




tabbed_interface.launch(server_name="0.0.0.0", server_port=1234,share=True)