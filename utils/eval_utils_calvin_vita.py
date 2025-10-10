from collections import defaultdict, namedtuple
import logging
import os, json, random
from pathlib import Path
import sys
import time
import PIL.Image as Image
import copy
from collections import deque
from moviepy.editor import ImageSequenceClip
from .data_utils_1 import prepare_data_vita
from  calvin.calvin_models.calvin_agent.models.calvin_base_model import CalvinBaseModel
import time
sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())
from calvin.calvin_models.calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin.calvin_models.calvin_agent.evaluation.utils import (
    collect_plan,
    count_success,
    create_tsne,
    get_env_state_for_initial_condition,
    get_log_dir,
    print_and_save,
)
import hydra
import numpy as np
from omegaconf import OmegaConf
from termcolor import colored
import torch
from tqdm.auto import tqdm
from calvin.calvin_env.calvin_env.envs.play_table_env import get_env
from utils.data_utils_1 import preprocess_image_vita
import functools
from utils.train_utils import get_cast_dtype
import cv2


os.environ['PYOPENGL_PLATFORM'] = 'egl'
logger = logging.getLogger(__name__)

EP_LEN = 360
NUM_SEQUENCES = 1000

def make_env(dataset_path):
    val_folder = Path(dataset_path) / "validation"
    env = get_env(val_folder, show_gui=False)

    return env

class ModelWrapper(CalvinBaseModel):
    def __init__(self, model, tokenizer, image_processor, cast_dtype, history_len=10, 
                calvin_eval_max_steps=360, action_pred_steps=3):
        super().__init__()
        self.model = model
        self.cast_type = cast_dtype
        self.use_diff = False
        self.tokenizer = tokenizer
        self.image_process_fn = functools.partial(preprocess_image_vita, image_processor=image_processor)
        self.action_hist_queue = []
        self.history_len = history_len
        self.calvin_eval_max_steps = calvin_eval_max_steps
        self.action_pred_steps = action_pred_steps
        self.device = "cuda"
        self.img_queue = deque(maxlen=history_len)
        self.gripper_queue = deque(maxlen=history_len)
        self.state_queue = deque(maxlen=history_len)
        self.mask_queue = deque(maxlen=history_len)
        self.text_queue = deque(maxlen=history_len)
        self.act_queue = deque(maxlen=history_len-1)

    def reset(self):
        self.img_queue = deque(maxlen=self.history_len)
        self.gripper_queue = deque(maxlen=self.history_len)
        self.state_queue = deque(maxlen=self.history_len)
        self.mask_queue = deque(maxlen=self.history_len)
        self.text_queue = deque(maxlen=self.history_len)
        self.act_queue = deque(maxlen=self.history_len-1)

    def step(self, obs, goal, timestep):
        # 获取当前步的输入
        image = obs["rgb_obs"]['rgb_static']
        image = Image.fromarray(image)
        image_x = self.image_process_fn([image])
        image_x = image_x.unsqueeze(1).to(dtype=self.cast_type)

        gripper = obs["rgb_obs"]['rgb_gripper']
        gripper = Image.fromarray(gripper)
        gripper = self.image_process_fn([gripper])
        gripper = gripper.unsqueeze(1).to(dtype=self.cast_type)
 
        text_x = goal # 对应的文本目标

        state = obs['robot_obs']
        state = torch.from_numpy(np.stack([state]))
        state = state.unsqueeze(1).to(dtype=self.cast_type)
        state = torch.cat([state[..., :6], state[..., [-1]]], dim=-1)


        with torch.no_grad():
            device = 'cuda'
            image_x = image_x.to(device)
            # text_x = text_x.to(device)
            gripper = gripper.to(device)
            state = state.to(device)
            self.img_queue.append(image_x)  
            self.gripper_queue.append(gripper)
            self.state_queue.append(state) 
            image_primary = torch.cat(list(self.img_queue), dim=1)
            image_wrist = torch.cat(list(self.gripper_queue), dim=1) 
            state = torch.cat(list(self.state_queue), dim=1) 
            num_step = image_primary.shape[1]
            
            if num_step < self.history_len:  
                input_image_primary = torch.cat([image_primary, image_primary[:, -1].
                repeat(1, self.history_len-num_step, 1, 1, 1)], dim=1)
                input_image_wrist = torch.cat([image_wrist, image_wrist[:, -1].
                repeat(1, self.history_len-num_step, 1, 1, 1)], dim=1)
                input_state = torch.cat([state, state[:, -1].repeat(1, self.history_len-num_step, 1)], dim=1)
            else:
                input_image_primary = image_primary
                input_image_wrist = image_wrist
                input_state = state
            
            batch_vita = prepare_data_vita(input_image_primary, input_image_wrist, goal, input_state, self.tokenizer)
            
            # batch_vita = prepare_data_vita(image_primary, image_wrist, goal, state, self.tokenizer)
             
            arm_action, gripper_action = self.model.module(
                input_ids = batch_vita['input_ids'],
                images = batch_vita['images'],
                attention_mask = batch_vita["attention_mask"],
                labels = batch_vita["labels"],
                states = batch_vita['states'],
                generate = True
            ) 
            
            arm_action = arm_action.view(arm_action.shape[0], -1, self.action_pred_steps, arm_action.shape[-1])
            gripper_action = gripper_action.view(gripper_action.shape[0], -1, 
                                                 self.action_pred_steps, gripper_action.shape[-1])
            
            action = torch.concat((arm_action[0, :, 1, :], gripper_action[0, :, 1, :] > 0.5), dim=-1)
            action[:, -1] = (action[:, -1] - 0.5) * 2  # scale to -1 or 1
            action = action.cpu().detach().to(dtype=torch.float16).numpy()
            action = action[-1]
                
        return action

from collections import Counter 
    
def evaluate_policy_ddp(model, env, epoch, calvin_conf_path, eval_log_dir=None, 
                        debug=False, create_plan_tsne=False, reset=False, diverse_inst=False, cur_time='00'):
    """
    Run this function to evaluate a model on the CALVIN challenge.

    Args:
        model: Must implement methods of CalvinBaseModel.
        env: (Wrapped) calvin env.
        epoch:
        eval_log_dir: Path where to log evaluation results. If None, logs to /tmp/evaluation/
        debug: If True, show camera view and debug info.
        create_plan_tsne: Collect data for TSNE plots of latent plans (does not work for your custom model)

    Returns:
        Dictionary with results
    """
    conf_dir = Path(calvin_conf_path)
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    
    # val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    if diverse_inst:
        with open('./utils/lang_annotation_cache.json', 'r') as f:
            val_annotations = json.load(f)
    else:
        val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    eval_log_dir = get_log_dir(eval_log_dir)
    
    # 包含了环境的初始状态，和对应的5个task
    
    with open('./utils/eval_sequences.json', 'r') as f:
        eval_sequences = json.load(f)
    device_num = int(torch.distributed.get_world_size())
    device_id = torch.distributed.get_rank()
    assert NUM_SEQUENCES % device_num == 0
    interval_len = int(NUM_SEQUENCES // device_num)
    eval_sequences = eval_sequences[device_id*interval_len:min((device_id+1)*interval_len, NUM_SEQUENCES)]
    results = []
    plans = defaultdict(list)
    local_sequence_i = 0
    base_sequence_i = device_id * interval_len

    # string_counter = Counter()
    
    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)
    
    # for initial_state, eval_sequence in eval_sequences:  
    #     # print(eval_sequence)
    #     for i in eval_sequence: 
    #         string_counter[i] += 1
    
    # for string, count in string_counter.items():
    #     print(f"String: {string}, Count: {count}")

    # # 将每个字符串及其出现次数按行写入到一个文本文件中
    # output_file_path = "string_counts_test.txt"
    # with open(output_file_path, "w") as file:
    #     for string, count in string_counter.items():
    #         file.write(f"{string}: {count}\n")

    # print(f"String counts have been written to {output_file_path}")
    # print(len(eval_sequences))   1000，只取前1000个
    for initial_state, eval_sequence in eval_sequences:
        # 一个初始状态和对应的5个task
        result = evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence, 
                                   val_annotations, plans, debug, eval_log_dir, 
                                   base_sequence_i+local_sequence_i, reset=reset, 
                                   diverse_inst=diverse_inst, cur_time=cur_time)
        # import pdb;pdb.set_trace()
        results.append(result)
        eval_sequences.set_description(
            " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]) + "|"
        )
        local_sequence_i += 1
    def merge_multi_list(res):
        tmp = []
        for l in res:
            tmp.extend(l)
        return tmp

    def extract_iter_from_tqdm(tqdm_iter):
        return [_ for _ in tqdm_iter]
    
    if create_plan_tsne:
        create_tsne(plans, eval_log_dir, epoch)

    eval_sequences = extract_iter_from_tqdm(eval_sequences)

    res_tup = [(res, eval_seq) for res, eval_seq in zip(results, eval_sequences)]
    all_res_tup = [copy.deepcopy(res_tup) for _ in range(device_num)] if torch.distributed.get_rank() == 0 else None
    torch.distributed.gather_object(res_tup, all_res_tup, dst=0)

    if torch.distributed.get_rank() == 0:
        res_tup_list = merge_multi_list(all_res_tup)
        res_list = [_[0] for _ in res_tup_list]
        eval_seq_list = [_[1] for _ in res_tup_list]
        print_and_save(res_list, eval_seq_list, eval_log_dir, epoch)

    return results

def evaluate_sequence(env, model, task_checker, initial_state, eval_sequence, val_annotations, 
                      plans, debug, eval_log_dir='', sequence_i=-1, reset=False, diverse_inst=False, cur_time='00'):
    """
    Evaluates a sequence of language instructions.
    """
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    success_counter = 0

    # print(eval_sequence, '****************') 5个task
    for subtask_i, subtask in enumerate(eval_sequence):
        # print(eval_sequence, subtask_i, subtask)
        # 取其中的一个task，给出对应的envs，env设置好了初始状态
        if reset:
            success = rollout(env, model, task_checker, subtask, val_annotations, plans, 
                              debug, eval_log_dir, subtask_i, sequence_i, diverse_inst=diverse_inst, 
                              robot_obs=robot_obs, scene_obs=scene_obs, cur_time = cur_time)
        else:
            success = rollout(env, model, task_checker, subtask, val_annotations, plans, debug, eval_log_dir, subtask_i, 
                              sequence_i, diverse_inst=diverse_inst, cur_time=cur_time)
        if success:
            # print(subtask, 'success')
            success_counter += 1
        else:
            # print(subtask, 'failed')
            return success_counter
    return success_counter


from .visualize_eval_seq import generate_single_seq_gif, plot_and_save_gifs

def as_gif(sequences, path, lang):
    # Render the images as the gif (15Hz control frequency): 
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)    
    # if len(sequences) < 360:
    imgs = generate_single_seq_gif(sequences, lang)
    plot_and_save_gifs(imgs, path)
    print('video has been saved in {}'.format(path))



def rollout(env, model, task_oracle, subtask, val_annotations, plans, debug, eval_log_dir='', 
            subtask_i=-1, sequence_i=-1, robot_obs=None, scene_obs=None, diverse_inst=False, cur_time='00'):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).
    """
    planned_actions = []
    if robot_obs is not None and scene_obs is not None:
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    obs = env.get_obs()
    # get lang annotation for subtask
    if diverse_inst:
        lang_annotation = val_annotations[sequence_i][subtask_i]
    else:
        lang_annotation = val_annotations[subtask][0]
    lang_annotation = lang_annotation.split('\n')[0]
    if '\u2019' in lang_annotation:
        lang_annotation.replace('\u2019', '\'')
        
    # print('lang_annotation', lang_annotation)
    # import pdb; pdb.set_trace()
    model.reset()
    start_info = env.get_info()

    
    sequences = []
    for step in range(EP_LEN):
        # print(step)
        action = model.step(obs, lang_annotation, step)
        if len(planned_actions) == 0:
            if action.shape == (7,):
                planned_actions.append(action)
            else:
                planned_actions.extend([action[i] for i in range(action.shape[0])])
        action = planned_actions.pop(0)
        obs, _, _, current_info = env.step(action)
        sequences.append(obs["rgb_obs"]["rgb_static"])
        if step == 0:
            collect_plan(model, plans, subtask)
        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            # as_gif(sequences=sequences, path=f"/videos_{cur_time}/{lang_annotation}.mp4", lang=lang_annotation)
            # print('success success success success success success success success ')
            return True
    
    # as_gif(sequences=sequences, path=f"/videos_{cur_time}/failed/{lang_annotation}.mp4", lang=lang_annotation)

    return False

def eval_one_epoch_calvin_ddp(args, model, dataset_path, image_processor, tokenizer, eval_log_dir=None, 
                              debug=False, future_act_len=-1, reset=False, diverse_inst=False, cur_time='00'):
    env = make_env(dataset_path)
    cast_dtype = get_cast_dtype(args.precision)
    # import pdb; pdb.set_trace()
    hist_len = args.sequence_length
     
    wrapped_model = ModelWrapper(
                        model, 
                        tokenizer, 
                        image_processor, 
                        cast_dtype, 
                        history_len=hist_len, 
                        calvin_eval_max_steps=EP_LEN,
                        action_pred_steps = args.action_pred_steps)
    evaluate_policy_ddp(wrapped_model, env, 0, args.calvin_conf_path, eval_log_dir=eval_log_dir, 
                        debug=debug, reset=reset, diverse_inst=diverse_inst, cur_time=cur_time)
