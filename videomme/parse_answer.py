import pandas as pd
import re
import os
import argparse

CATEGORY_DICT = {
"meishi": "美食",
"lvxing": "旅行",
"lanqiu": "篮球",
"tianwen": "天文",
"zuqiu": "足球",
"shengwu": "生物医学",
"wutaiju": "舞台剧",

"shishang": "时尚",
"caijing": "财经商业",
"keji": "科技数码",

"renwen": "人文历史",
"wenxue": "文学艺术",
"dili": "地理",
"xinwen": "新闻",
"jilupian": "纪录片",
"zongyi": "综艺",
"dianying": "电影剧集",
"mengchong": "萌宠",
"youxi": "游戏电竞",
"donghua": "动画",

"shenghuo": "生活",
"moshu": "魔术",
"zaji": "杂技特效",
"shougong": "手工教程",
"qita": "其他",

"falv": "法律",
"tianjing": "田径",
"richang": "日常",
"yundong": "运动",

"duoyuzhong": "多语种"

}


VIDEO_TYPE_DICT = {
"s": "短视频 <= 2 min", 
"m": "中视频 4-15 min", 
"l": "长视频 30-60 min"
}

def extract_characters_regex(s):
    if s is None or str(s) == "nan":
        return ""

    s = s.replace("Answer:", "")
    s = s.replace("The correct answer is", "")
    s = s.replace("The answer is", "")

    matches = re.search(r'[ABCD]', s)
    if matches is None or matches[0][0] not in ["A", "B", "C", "D"]:
        print(f"Invalid answer: {s}")
        return ""
    return matches[0][0].upper() 


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_types", type=str, default="s,m,l", help="Comma separated list of video types")
    parser.add_argument("--result_dir", type=str, default="qa_wo_sub_revision", help="Directory containing the results")
    return parser.parse_args()

args = parse_args()
video_types = args.video_types.split(",")
result_dir = args.result_dir


overall_correct = 0
overall_total = 0

for video_type in video_types:

    num_correct = 0
    num_total = 0
    for category in CATEGORY_DICT.keys():

        if not os.path.exists(f"{result_dir}/{video_type}/{category}.csv"):
            print(f"{result_dir}/{video_type}/{category}.csv does not exist")
            continue

        num_total += 30
        cate_df = pd.read_csv(f"{result_dir}/{video_type}/{category}.csv")
        correct = 0
        for (cate_id, cate_row) in cate_df.iterrows():

            gt = cate_row[["答案一", "答案二", "答案三"]]
            pred = cate_row[["模型回答一", "模型回答二", "模型回答三"]]

            pred = pred.apply(extract_characters_regex)

            correct += (gt.to_numpy() == pred.to_numpy()).sum()
            
        num_correct += correct

    print(f"Video Type {video_type}: {num_correct / num_total * 100:.2f}%")

    overall_correct += num_correct
    overall_total += num_total

print(f"Overall: {overall_correct / overall_total * 100:.2f}%")



