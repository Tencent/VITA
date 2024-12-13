import pandas as pd
import cv2
import pysubs2
import torch
import os
from PIL import Image
from decord import VideoReader, cpu
import numpy as np
import argparse
import time
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from vita.util.data_utils_video_audio_neg_patch import dynamic_preprocess
from vita.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, MAX_IMAGE_LENGTH, DEFAULT_VIDEO_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_AUDIO_TOKEN
from vita.conversation import conv_templates, SeparatorStyle
from vita.model.builder import load_pretrained_model
from vita.util.mm_utils import tokenizer_image_token, tokenizer_image_audio_token, get_model_name_from_path, KeywordsStoppingCriteria
from vita.util.utils import disable_torch_init


VIDEO_TYPE_DICT = {
"s": "短视频 <= 2 min", 
"m": "中视频 4-15 min", 
"l": "长视频 30-60 min"
}

CATEGORY_DICT = {
"meishi": "美食",
"lvxing": "旅行",
"shishang": "时尚",
"lanqiu": "篮球",
"caijing": "财经商业",
"keji": "科技数码",
"zuqiu": "足球",
"tianwen": "天文",
"shengwu": "生物医学",
"wutaiju": "舞台剧",
"falv": "法律",
"shenghuo": "生活",
"moshu": "魔术",
"zaji": "杂技特效",
"shougong": "手工教程",
"xinwen": "新闻",
"jilupian": "纪录片",
"zongyi": "综艺",
"dianying": "电影剧集",
"mengchong": "萌宠",
"youxi": "游戏电竞",
"donghua": "动画",
"renwen": "人文历史",
"wenxue": "文学艺术",
"dili": "地理",
"tianjing": "田径",
"richang": "日常",
"yundong": "运动",
"qita": "其他",
"duoyuzhong": "多语种"
}

REPONSIBLE_DICT = {
    "lyd": ["meishi", "lvxing", "lanqiu", "tianwen"],
    "jyg": ["zuqiu", "shengwu", "wutaiju"],
    "wzh": ["shishang", "caijing", "keji", "duoyuzhong"],
    "wzz": ["renwen", "wenxue", "dili", "qita"],
    "zcy": ["xinwen", "jilupian", "zongyi", "dianying"],
    "by": ["mengchong", "youxi", "donghua"],
    "dyh": ["shenghuo", "moshu", "zaji", "shougong"],
    "lfy": ["falv", "tianjing", "richang", "yundong"]
}

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")

    parser.add_argument("--responsible_man", type=str, default="dyh", help="Category of the video")
    parser.add_argument("--categories", type=str, default=None, help="Category of the video")
    parser.add_argument("--video_type", type=str, default="s", help="Type of the video. Choose from ['s', m', 'l']")
    parser.add_argument("--video_dir", type=str, default="../yt-videos/", help="Output path")
    parser.add_argument("--use_subtitles", action='store_true', help="Use subtitles")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--model-path", type=str, default="../../models/Chat-UniVi-7B-v1.5")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model_type", type=str, default='mixtral-8x7b')
    parser.add_argument("--conv_mode", type=str, default='mixtral_two')
    parser.add_argument("--output_dir", type=str, default="qa_wo_sub")
    
    args = parser.parse_args()
    return args

def get_seq_frames(total_num_frames, desired_num_frames):
    """
    Calculate the indices of frames to extract from a video.

    Parameters:
    total_num_frames (int): Total number of frames in the video.
    desired_num_frames (int): Desired number of frames to extract.

    Returns:
    list: List of indices of frames to extract.
    """

    # Calculate the size of each segment from which a frame will be extracted
    seg_size = float(total_num_frames - 1) / desired_num_frames

    seq = []
    for i in range(desired_num_frames):
        # Calculate the start and end indices of each segment
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))

        # Append the middle index of the segment to the list
        seq.append((start + end) // 2)

    return seq


def _get_rawvideo_dec_decord(video_path, image_processor, max_frames=MAX_IMAGE_LENGTH, image_resolution=384, video_framerate=1, s=None, e=None, image_aspect_ratio='square'):
    # speed up video decode via decord.

    if s is None:
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
        print(video_path)
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
        else:
            sample_pos = all_pos

        patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]

        if image_aspect_ratio == 'pad':
            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            patch_images = [expand2square(i, tuple(int(x * 255) for x in image_processor.image_mean)) for i in patch_images]
            patch_images = [image_processor.preprocess(i, return_tensors='pt')['pixel_values'][0] for i in patch_images]
        else:
            patch_images = [image_processor.preprocess(i, return_tensors='pt')['pixel_values'][0] for i in patch_images]

        patch_images = torch.stack(patch_images)
        slice_len = patch_images.shape[0]

        return patch_images, slice_len, sample_pos
    else:
        print("video path: {} error.".format(video_path))

def _get_rawvideo_dec(video_path, image_processor, max_frames=MAX_IMAGE_LENGTH, image_resolution=224, video_framerate=1, s=None, e=None, image_aspect_ratio='square'):
    # speed up video decode via decord.
    video_mask = np.zeros(max_frames, dtype=np.int64)
    max_video_length = 0

    # T x 3 x H x W
    video = np.zeros((max_frames, 3, image_resolution, image_resolution), dtype=np.float64)

    if s is None:
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
        cv2_vr = cv2.VideoCapture(video_path)

    else:
        raise FileNotFoundError

    fps = cv2_vr.get(cv2.CAP_PROP_FPS)
    duration = int(cv2_vr.get(cv2.CAP_PROP_FRAME_COUNT))

    f_start = 0 if start_time is None else int(start_time * fps)
    f_end = int(min(1000000000 if end_time is None else end_time * fps, duration - 1))
    num_frames = f_end - f_start + 1
    if num_frames > 0:
        # T x 3 x H x W
        sample_fps = int(video_framerate)
        t_stride = int(round(float(fps) / sample_fps))

        all_pos = list(range(f_start, f_end + 1, t_stride))
        if len(all_pos) > max_frames:
            sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=max_frames, dtype=int)]
        else:
            sample_pos = all_pos

        patch_images = []
        count = 0

        while cv2_vr.isOpened():
            success, frame = cv2_vr.read()
            if not success:
                break
            if count in sample_pos:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                patch_images.append(torch.from_numpy(frame).permute(2, 0, 1))
            count += 1
        cv2_vr.release()

        patch_images = torch.stack([image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0] for img in patch_images])
        slice_len = patch_images.shape[0]

        max_video_length = max_video_length if max_video_length > slice_len else slice_len
        if slice_len < 1:
            pass
        else:
            video[:slice_len, ...] = patch_images

        return patch_images, slice_len, sample_pos
    else:
        print("video path: {} error.".format(video_path))

    video_mask[:max_video_length] = [1] * max_video_length

    return torch.from_numpy(video), video_mask


def main(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    model_base = args.model_base
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name, args.model_type)

    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    image_processor = vision_tower.image_processor

    audio_encoder = model.get_audio_encoder()
    #audio_encoder.to(device="cuda", dtype=torch.float16)
    audio_encoder.to(dtype=torch.float16)
    audio_processor = audio_encoder.audio_processor

    model.eval()
    audio = torch.zeros(400, 80)
    audio_length = audio.shape[0]
    audio = torch.unsqueeze(audio, dim=0)
    audio_length = torch.unsqueeze(torch.tensor(audio_length), dim=0)
    audios = dict()
    audios['audios'] = audio.half().cuda()
    audios['lengths'] = audio_length.half().cuda()


    video_types = args.video_type.split(",")
    video_dir = args.video_dir

    if args.categories is not None:
        categories = args.categories.split(",")
    else:
        categories = REPONSIBLE_DICT[args.responsible_man]

    df = pd.read_csv(f'{video_dir}/qa_annotations.csv')
    df["模型回答一"] = None
    df["模型回答二"] = None
    df["模型回答三"] = None


    for video_type in video_types:
        for category in categories:

            print(f"Processing category: {category}, video type: {video_type}")

            output_dir = f'{args.output_dir}/{video_type}'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            condition1 = (df["子任务"] == CATEGORY_DICT[category])
            condition2 = (df["时长类别"] == VIDEO_TYPE_DICT[video_type])
            filtered_rows = df[condition1 & condition2]

            for idx, row in filtered_rows.iterrows():

                indices = ["一", "二", "三"]

                # Get the video name and questions
                video_name = row["视频命名"]
                if isinstance(video_name, float):
                    continue

                questions = [row[f"问题{_}"] for _ in indices]

                video_file_name = f"{video_dir}/{category}/{video_type}/video/{video_name}"

                # Get the subtitles file name
                for sub in os.listdir(f"{video_dir}/{category}/{video_type}/subtitles"):
                    subtitles_file_name = ""
                    if video_name[:-4] in sub and sub.endswith(".srt"):
                        subtitles_file_name = f"{video_dir}/{category}/{video_type}/subtitles/{sub}"
                        break

                # Check if the video file exists
                if not os.path.exists(video_file_name):
                    print(f"No {video_file_name}.")
                    continue

  
                import pdb; pdb.set_trace() 
                if args.max_frames:
                    video_frames, slice_len, selected_frame_ids = _get_rawvideo_dec(video_file_name, image_processor, max_frames=args.max_frames, video_framerate=1, image_resolution=448, image_aspect_ratio=getattr(model.config, "image_aspect_ratio", None))
                else:
                    video_frames, slice_len, selected_frame_ids = _get_rawvideo_dec(video_file_name, image_processor, max_frames=MAX_IMAGE_LENGTH, video_framerate=1, image_resolution=448, image_aspect_ratio=getattr(model.config, "image_aspect_ratio", None))

                if args.use_subtitles and os.path.exists(subtitles_file_name):

                    subtitles = []
                    cv2_vr = cv2.VideoCapture(video_file_name)
                    duration = int(cv2_vr.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cv2_vr.get(cv2.CAP_PROP_FPS)
                    # vreader = VideoReader(video_file_name, ctx=cpu(0))
                    # fps = vreader.get_avg_fps()

                    subs = pysubs2.load(subtitles_file_name, encoding="utf-8")

                    for selected_frame_id in selected_frame_ids:
                        sub_text = ""
                        for sub in subs:
                            if sub.start < pysubs2.make_time(frames=selected_frame_id, fps=fps) and sub.end > pysubs2.make_time(frames=selected_frame_id, fps=fps):
                                sub_text = sub.text.replace("\\N", " ")
                                break
                        if sub_text.strip() != "":
                            subtitles.append(sub_text) 
                    cv2_vr.release()

                    subtitles = "This video's subtitles are listed below: \n" + "\n".join(subtitles) + "\n"
                else:
                    subtitles = ""
    

                start_time = time.time()
                for id, question in zip(indices, questions):
                    question = subtitles + question + "Answer with the option's letter from the given choices directly."
                    question = DEFAULT_IMAGE_TOKEN * slice_len + '\n' + question

                    conv = conv_templates[args.conv_mode].copy()
                    conv.append_message(conv.roles[0], question)
                    conv.append_message(conv.roles[1], None)
                    modality='video'
                    prompt = conv.get_prompt(modality)

                    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
                        0).cuda()

                    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                    keywords = [stop_str]
                    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

                    with torch.inference_mode():
                        output_ids = model.generate(
                            input_ids,
                            images=video_frames.half().cuda(),
                            audios=audios,
                            do_sample=True,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            num_beams=args.num_beams,
                            output_scores=True,
                            return_dict_in_generate=True,
                            max_new_tokens=10,
                            use_cache=True,
                            stopping_criteria=[stopping_criteria]
                        )

                    output_ids = output_ids.sequences
                    input_token_len = input_ids.shape[1]
                    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
                    if n_diff_input_output > 0:
                        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
                    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
                    outputs = outputs.strip()
                    if outputs.endswith(stop_str):
                        outputs = outputs[:-len(stop_str)]
                    outputs = outputs.strip()

                    filtered_rows.loc[idx, f"模型回答{id}"] = outputs

                    print(f"{prompt}\nAnswer: {outputs}\n")
                    
                    print(f"Time taken to generate answers: {time.time() - start_time:.2f} seconds\n")
                    

            # Save the updated dataframe to the excel file
            results = filtered_rows[["子任务", "视频命名", "问题一", "模型回答一", "答案一", "问题二", "模型回答二", "答案二", "问题三", "模型回答三", "答案三"]]
                
            results.to_csv(f'{output_dir}/{category}.csv', index=False)
        

if __name__ == '__main__':
    args = parse_args()
    main(args)


