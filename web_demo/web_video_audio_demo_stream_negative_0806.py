
import torch
import os
from bunny.constants import DEFAULT_AUDIO_TOKEN, DEFAULT_IMAGE_TOKEN, MAX_IMAGE_LENGTH, IMAGE_TOKEN_INDEX, MIN_IMAGE_LENGTH
from bunny.conversation import conv_templates, SeparatorStyle
from bunny.util.mm_utils import tokenizer_image_token, tokenizer_image_audio_token ,dynamic_preprocess
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from decord import VideoReader, cpu
import numpy as np
import copy
import gradio as gr
import os
import re
import torchaudio
import json
import base64
import io
from scipy.io import wavfile
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.tts.v20190823 import tts_client, models
from vllm import LLM, SamplingParams
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoFeatureExtractor


PUNCTUATION = "！？。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."


import re


# 数字到汉字的映射
num_to_chinese = {
    '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
    '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'
}

def remove_punctuation_and_convert_numbers(input_str):
    input_str =  input_str.replace('</s>', '').replace('<2>', '').replace('<1>', '').replace('<3>', '').replace('<unk>', '').replace('\n','')

    # 将数字转换为汉字
    result = []
    for char in input_str:
        if char.isdigit():
            result.append(num_to_chinese[char])
        else:
            result.append(char)
    output_str = ''.join(result)
    return output_str


 



import math
from numba import jit


@jit
def float_to_int16(audio: np.ndarray) -> np.ndarray:
    am = int(math.ceil(float(np.abs(audio).max())) * 32768)
    am = 32767 * 32768 // am
    return np.multiply(audio, am).astype(np.int16)



def is_video(file_path):
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv','.webm'}
    _, ext = os.path.splitext(file_path)
    return ext.lower() in video_extensions


def is_wav(file_path):
    video_extensions = {'.wav'}
    _, ext = os.path.splitext(file_path)
    return ext.lower() in video_extensions

def escape_markdown(text):
    # List of Markdown special characters that need to be escaped
    md_chars = ['<', '>']

    # Escape each special character
    for char in md_chars:
        text = text.replace(char, '\\' + char)

    return text

# import ffmpeg
import cv2
def convert_webm_to_mp4(input_file, output_file):
    try:
        cap = cv2.VideoCapture(input_file)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        cap.release()
        out.release()
    except Exception as e:
        print(f"Error: {e}")
        raise

def is_image(file_path):
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
    _, ext = os.path.splitext(file_path)
    return ext.lower() in image_extensions


# 自定义深拷贝函数
def custom_deepcopy(obj):
    if isinstance(obj, list):
        return [custom_deepcopy(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: custom_deepcopy(value) for key, value in obj.items()}
    else:
        try:
            return copy.deepcopy(obj)
        except TypeError:
            return obj  # 跳过不可序列化的对象
        

def _get_rawvideo_dec(video_path, max_frames=MAX_IMAGE_LENGTH, min_frames=MIN_IMAGE_LENGTH, video_framerate=1, s=None, e=None):
    # speed up video decode via decord.

    if s is None or e is None:
        start_time, end_time = None, None
    else:
        start_time = int(s)
        end_time = int(e)
        start_time = start_time if start_time >= 0. else 0.
        end_time = end_time if end_time >= 0. else 0.
        if start_time > end_time:
            start_time, end_time = end_time, start_time
        elif start_time == end_time:
            end_time = start_time + 1

    if os.path.exists(video_path):
        vreader = VideoReader(video_path, ctx=cpu(0))
    else:
        raise FileNotFoundError

    fps = vreader.get_avg_fps()
    f_start = 0 if start_time is None else int(start_time * fps)
    f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
    num_frames = f_end - f_start + 1
    if num_frames > 0:
        # T x 3 x H x W
        sample_fps = int(video_framerate)
        t_stride = int(round(float(fps) / sample_fps))

        all_pos = list(range(f_start, f_end + 1, t_stride))
        if len(all_pos) > max_frames:
            sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=max_frames, dtype=int)]
        elif len(all_pos) < min_frames:
            sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=min_frames, dtype=int)]
        else:
            sample_pos = all_pos

        patch_images = [Image.fromarray(f).convert("RGB") for f in vreader.get_batch(sample_pos).asnumpy()]

        return patch_images,len(patch_images)
    else:
        print("video path: {} error.".format(video_path))




def _parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split("`")
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f"<br></code></pre>"
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", r"\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text

def _remove_image_special(text):
    text = text.replace('<ref>', '').replace('</ref>', '')
    return re.sub(r'<box>.*?(</box>|$)', '', text)

def _launch_demo(llm,sampling_params,tokenizer, feature_extractor, client, req):

    def predict(_chatbot, task_history):




        chat_query = task_history[-1][0]

        print(task_history)

        conv_mode = "mixtral_two"


        conv = conv_templates[conv_mode].copy()
        

        # 用于控制当轮是否是第一轮对话，为true代表已经添加过图片，即不是第一轮，

        
        all_audio_path = []
        # import pdb;pdb.set_trace()
        
        all_visual_tensor = []

        qs = ''
        input_mode = 'lang'
        for i, (q, a) in enumerate(task_history):
            if isinstance(q, (tuple, list)):
                #文件
                # import pdb; pdb.set_trace()
                if is_image(q[0]):
                    image = [Image.open(q[0]).convert("RGB")]
                    images = dynamic_preprocess(image, image_size=448,
                                    use_thumbnail=True ,
                                    max_num= 12 )
                    import pdb;pdb.set_trace()
                    # image_tensor = load_image(q[0], 448,dymatic=True).to(dtype=torch.float16, device="cuda")
                    # print(image_tensor.shape)
                    all_visual_tensor.extend(images)
                    input_mode = 'image'
                    qs += DEFAULT_IMAGE_TOKEN * len(image) + '\n'

                elif is_video(q[0]):


                    #没按照训练的来，先按yuhan的替换
             
                    video_frames, slice_len =_get_rawvideo_dec(q[0])

                    all_visual_tensor.extend(video_frames)
                    input_mode = 'video'
                    qs += DEFAULT_IMAGE_TOKEN * slice_len  + '\n'    

                elif is_wav(q[0]):
                    if a != None and a.startswith('<2>'):
                        continue
                    else:
                        all_audio_path.append(q[0])
                        
                        new_q = qs + DEFAULT_AUDIO_TOKEN
                        qs = ''
                        
                        conv.append_message(conv.roles[0], new_q)
                        conv.append_message(conv.roles[1], a)
            else:
                #文本
                # import pdb;pdb.set_trace()
               
                new_q = qs + q
                qs = ''
              
                conv.append_message(conv.roles[0], new_q)
                conv.append_message(conv.roles[1], a)

                # 处理文本 ,有可能是之前的文本，也有可能是新的文本

        print(conv)
        prompt = conv.get_prompt(input_mode)
        # import pdb;pdb.set_trace()


        if all_audio_path != []:
            input_ids = tokenizer_image_audio_token(prompt, tokenizer, IMAGE_TOKEN_INDEX)
        else:
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX)
        
        

        if all_audio_path != []:
            audio_list = []
            for single_audio_path in all_audio_path:
                try:
                    audio, sr = torchaudio.load(single_audio_path)
                    audio_features = feature_extractor(audio, sampling_rate=sr, return_tensors="pt")["input_features"]

                    
                    audio_list.append(audio_features.squeeze(0))
          
                except Exception as e:
                    print(f"Error processing {single_audio_path}: {e}")
            

            else:
                print("No valid audio files were processed.")


        # import pdb;pdb.set_trace()
        sampling_params = SamplingParams(temperature=0.01, max_tokens=1024, best_of=1, skip_special_tokens=False)



        if all_visual_tensor == [] and all_audio_path == []:
            datapromt={
                 "prompt_token_ids": input_ids,
            }
 
        elif all_visual_tensor != [] and all_audio_path == []:
            datapromt={
                "prompt_token_ids": input_ids,
                "multi_modal_data": {"image": all_visual_tensor,
                                    },
            }
        elif all_visual_tensor == [] and all_audio_path != []:
            datapromt={
                "prompt_token_ids": input_ids,
                "multi_modal_data": {"audio": audio_list,
                                     },
            }
        else:
            datapromt={
                "prompt_token_ids": input_ids,
                "multi_modal_data": {"image": all_visual_tensor,
                                     "audio": audio_list,
                                     },
            }
        # import pdb;pdb.set_trace()
        output = llm.generate(
                datapromt,
                sampling_params=sampling_params
        )
        outputs = output[0].outputs[0].text
        # import pdb;pdb.set_trace()

        # import pdb;pdb.set_trace()
        # if not (prompt_id.cpu() == torch.tensor([51497])).item():
        task_history[-1] = (chat_query, outputs)
        vad_input = remove_punctuation_and_convert_numbers(outputs)


        print(vad_input)
  
        _chatbot[-1] = (chat_query, _remove_image_special(_parse_text(vad_input)))



        print("query",chat_query)
        print("task_history",task_history)
        print(_chatbot)
        print("answer:  ",outputs)
        yield _chatbot














    def add_text(history, task_history, text):
        task_text = text
        if len(text) >= 2 and text[-1] in PUNCTUATION and text[-2] not in PUNCTUATION:
            task_text = text[:-1]
        history = history + [(_parse_text(text), None)]
        task_history = task_history + [(task_text, None)]
        return history, task_history, ""

    def add_file(history, task_history, file):
        history = history + [((file.name,), None)]
        task_history = task_history + [((file.name,), None)]

        return history, task_history


    
    def add_audio(history, task_history, file):
        print(file)

        if file is None:
            return history, task_history
        history = history + [((file,), None)]
        task_history = task_history + [((file,), None)]
        return history, task_history


    def add_video(history, task_history, file):
            print(file)
            if file is None:
                return history, task_history
            new_file_name=file.replace(".webm",".mp4")
            if file.endswith(".webm"):
                convert_webm_to_mp4(file,new_file_name)
            task_history = task_history + [((new_file_name,), None)]
            return history, task_history




    def text_to_audio(history ,use_tts):

        if use_tts == 'true':
            #use tecentcloud api,you cloud change it for other tts
            text = history[-1][1]
            params = {
            "Text": text,
            "SessionId": "session-1234",
            "Volume": 1,
            "Speed": 0,
            "ProjectId": 0,
            "ModelType": 1,
            "VoiceType": 301009,
            "PrimaryLanguage": 1,
            "SampleRate": 16000,
            "Codec": "wav",
            "EnableSubtitle": True
            }
            req.from_json_string(json.dumps(params))
            resp = client.TextToVoice(req)
            base64_audio_data = json.loads(resp.to_json_string())['Audio']
            audio_data = base64.b64decode(base64_audio_data)
            audio_stream = io.BytesIO(audio_data)
            sample_rate, audio_array = wavfile.read(audio_stream)
            
            # import pdb;pdb.set_trace()
            yield sample_rate, audio_array
        else:
            yield None,None








    def reset_user_input():
        return gr.update(value="")

    def reset_state(task_history):
        task_history.clear()
        return []

    with gr.Blocks(title="VideoMLLM") as demo:

        gr.Markdown("""<center><font size=8>VITA</center>""")
        chatbot = gr.Chatbot(label='VITA', elem_classes="control-height", height=500)
        query = gr.Textbox(lines=2, label='Text Input')
        task_history = gr.State([])
        with gr.Row():
            add_text_button = gr.Button("提交文本")
            add_file_button = gr.Button("上传音频")
        with gr.Row():
            with gr.Column(scale=2):
                addfile_btn = gr.UploadButton("📁 Upload (上传文件[视频,图片])", file_types=["video", "image"])
                video_input = gr.Video(sources=[ "webcam"], height=400, width=700, container=True, interactive=True, show_download_button=True,label="📹 录制视频")
   
        
            with gr.Column(scale=1):
                empty_bin = gr.Button("🧹 Clear History (清除历史)")
                record_btn = gr.Audio(sources=[ "microphone","upload"], type="filepath", label="🎤 Record (录音)",show_download_button=True, waveform_options=gr.WaveformOptions(sample_rate=16000))
                use_tts = gr.Radio(["true", "false"], value='true',label="use_tts")
                audio_output = gr.Audio(
                    label="Output Audio",
                    value=None,
                    format= "wav",
                    autoplay=True,
                    streaming=True,
                    interactive=False,
                    show_label=True,
                    waveform_options=gr.WaveformOptions(
                        sample_rate=16000,
                    ),
                )

  
        add_text_button.click(add_text, [chatbot, task_history, query], [chatbot, task_history], show_progress=True).then(
            reset_user_input, [], [query]
        ).then(
                predict, [chatbot, task_history], [chatbot], show_progress=True    #这里先返回实时语音
        ).then(
            text_to_audio, [chatbot,use_tts], [audio_output], show_progress=True
        )


        video_input.stop_recording(add_video, [chatbot, task_history, video_input], [chatbot, task_history], show_progress=True)
 
        empty_bin.click(reset_state, [task_history], [chatbot], show_progress=True)
        # regen_btn.click(regenerate, [chatbot, task_history], [chatbot], show_progress=True)
        addfile_btn.upload(add_file, [chatbot, task_history, addfile_btn], [chatbot, task_history], show_progress=True)



        add_file_button.click(add_audio, [chatbot, task_history,record_btn], [chatbot, task_history], show_progress=True
        ).then(
                predict, [chatbot, task_history], [chatbot], show_progress=True    #这里先返回实时语音
        ).then(
            text_to_audio, [chatbot,use_tts], [audio_output], show_progress=True
        )
        


    server_port = 18805
    demo.launch(
        share=False,
        debug=True,
        server_name="0.0.0.0",
        server_port=server_port,
        show_api=False,
        show_error=False,
        auth=('123','123'),
        )

def main():



  
    model_path='checkpoint-1800'
  
    print(torch.cuda.device_count())
    llm = LLM(
        model=model_path,
        dtype="float16",
        tensor_parallel_size=2,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        disable_custom_all_reduce=True,
    )  

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.01, max_tokens=100, best_of=1, skip_special_tokens=False)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # model, tokenizer, image_processor,audio_processor=None,None,None,None
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path, subfolder="feature_extractor", trust_remote_code=True)




    httpProfile = HttpProfile()
    httpProfile.endpoint = "tts.tencentcloudapi.com"

    # 实例化一个client选项，可选的，没有特殊需求可以跳过
    cred = credential.Credential("", "")
    clientProfile = ClientProfile()
    clientProfile.httpProfile = httpProfile
    # 实例化要请求产品的client对象,clientProfile是可选的
    client = tts_client.TtsClient(cred, "ap-shanghai", clientProfile)

    # 实例化一个请求对象,每个接口都会对应一个request对象
    req = models.TextToVoiceRequest()



    _launch_demo( llm,sampling_params,tokenizer, feature_extractor, client, req)


if __name__ == '__main__':
    main()
