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
from glob import glob
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


def load_video(video_path, image_processor, num_frames=MAX_IMAGE_LENGTH, image_aspect_ratio='square'):
    fps = 1 
    video_frame_list = sorted(glob(os.path.join(video_path, "*.png")))
    frame_idx = get_seq_frames(len(video_frame_list), min(num_frames, len(video_frame_list)))
    selected_frames = [video_frame_list[idx] for idx in frame_idx]

    patch_images = [Image.open(img_file).convert('RGB') for img_file in selected_frames]
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
    assert len(frame_idx) == slice_len

    video_info = {
        "fps": fps,
        "duration": len(video_frame_list),
        "num_frames": len(frame_idx),
        "selected_frame_ids": frame_idx
    }
    return patch_images, video_info


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
    audio_for_llm_lens = 60
    audio = torch.unsqueeze(audio, dim=0)
    audio_length = torch.unsqueeze(torch.tensor(audio_length), dim=0)
    audio_for_llm_lens = torch.unsqueeze(torch.tensor(audio_for_llm_lens), dim=0)
    audios = dict()
    audios['audios'] = audio.half().cuda()
    audios['lengths'] = audio_length.half().cuda()
    audios["lengths_for_llm"] = audio_for_llm_lens.cuda()

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

                # video_file_name = f"{video_dir}/{category}/{video_type}/video/{video_name}"
                # # Check if the video file exists
                # if not os.path.exists(video_file_name):
                #     print(f"No {video_file_name}.")
                #     continue

                # Get the subtitles file name
                for sub in os.listdir(f"{video_dir}/{category}/{video_type}/subtitles"):
                    subtitles_file_name = ""
                    if video_name[:-4] in sub and sub.endswith(".srt"):
                        subtitles_file_name = f"{video_dir}/{category}/{video_type}/subtitles/{sub}"
                        break

                video_file_name = f"{video_dir}/{category}/{video_type}/video/{video_name}"
                if os.path.exists(video_file_name):
                    video_name, _ = os.path.splitext(video_name)
                    video_file_name = f"{video_dir}/extracted_frames/{video_name}"
                    if args.max_frames:
                        video_frames, video_info = load_video(video_file_name, image_processor, args.max_frames, image_aspect_ratio=getattr(model.config, "image_aspect_ratio", None))
                    else:
                        video_frames, video_info = load_video(video_file_name, image_processor, MAX_IMAGE_LENGTH, image_aspect_ratio=getattr(model.config, "image_aspect_ratio", None))
                else:
                    print(f"No {video_file_name}."); continue

                if args.use_subtitles and os.path.exists(subtitles_file_name):
                    subtitles = []
                    subs = pysubs2.load(subtitles_file_name, encoding="utf-8")

                    for selected_frame_id in video_info["selected_frame_ids"]:
                        sub_text = ""
                        for sub in subs:
                            cur_time = pysubs2.make_time(frames=selected_frame_id, fps=video_info["fps"])
                            if sub.start < cur_time and sub.end > cur_time:
                                sub_text = sub.text.replace("\\N", " ")
                                break
                        subtitles.append(sub_text)
                    subtitles = "This video's subtitles are listed below: \n" + "\n".join(subtitles) + "\n"
                else:
                    subtitles = ""

                start_time = time.time()
                for id, question in zip(indices, questions):
                    #question = subtitles + question + "Answer with the option's letter from the given choices directly."
                    question = subtitles + question + "Please respond with only the letter of the correct answer."
                    question = DEFAULT_IMAGE_TOKEN * len(video_frames) + '\n' + question

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

#                    sf_masks = torch.tensor([1]*len(video_frames)).cuda()
                    sf_masks = None
                    with torch.inference_mode():
                        output_ids = model.generate(
                            input_ids,
                            images=video_frames.half().cuda(),
                            audios=audios,
                            sf_masks=sf_masks,
                            do_sample=True,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            num_beams=args.num_beams,
                            output_scores=True,
                            return_dict_in_generate=True,
                            max_new_tokens=10,
                            use_cache=True,
                            stopping_criteria=[stopping_criteria],
                            shared_v_pid_stride=None#2#16#8#4#1#None,
                        )

                    output_ids = output_ids.sequences
                    input_token_len = input_ids.shape[1]

                    if args.model_type == "mixtral-8x7b":
                        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
                        if n_diff_input_output > 0:
                            print(f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids")
                            output_ids = output_ids[:, input_token_len:]
                    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

                    outputs = outputs.strip()
                    if outputs.endswith(stop_str):
                        outputs = outputs[:-len(stop_str)]
                    outputs = outputs.strip()
                    if '☞' in outputs or '☜' in outputs or '☟' in outputs:
                        outputs = outputs[1:]

                    filtered_rows.loc[idx, f"模型回答{id}"] = outputs

                    print(f"{prompt}\nAnswer: {outputs}\n")
                    
                    print(f"Time taken to generate answers: {time.time() - start_time:.2f} seconds\n")
                    

            # Save the updated dataframe to the excel file
            results = filtered_rows[["子任务", "视频命名", "问题一", "模型回答一", "答案一", "问题二", "模型回答二", "答案二", "问题三", "模型回答三", "答案三"]]
                
            results.to_csv(f'{output_dir}/{category}.csv', index=False)
        

if __name__ == '__main__':
    args = parse_args()
    main(args)


