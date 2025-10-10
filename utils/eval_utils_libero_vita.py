import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ['MUJOCO_GL'] = 'osmesa'

from pathlib import Path
import copy
import io
import distutils.dir_util
import numpy as np
import time
import torch
from torch.distributed import gather
from collections import deque
import functools
from scipy.spatial.transform import Rotation as R
from tqdm.auto import tqdm

from utils.train_utils_1 import get_cast_dtype
from utils.data_utils_1 import prepare_data_vita, preprocess_image_vita

# libero
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image
from pdb import set_trace 

INDEX = 0  # 这是一个全局变量

def increment():
    global INDEX  # 声明要用全局变量
    INDEX += 1
    return INDEX


def quaternion_to_euler(q):
    rot = R.from_quat(q)
    euler = rot.as_euler('xyz', degrees=False)
    
    return euler


def save_tensor_as_png(img_tensor, filename):
    """
    将PyTorch张量保存为PNG图片。
    支持输入为 [C, H, W] 或 [H, W, C]，支持GPU和bfloat16等类型。
    """
    # 1. 移到CPU并转为float32
    img = img_tensor.detach().to(torch.float32).cpu()
    
    # 2. 调整形状为 [H, W, C]
    if img.ndim == 3:
        if img.shape[0] == 3 or img.shape[0] == 1:  # [C, H, W]
            img = img.permute(1, 2, 0)  # [H, W, C]
        # 否则假设已经是 [H, W, C]
    elif img.ndim == 2:
        img = img.unsqueeze(-1)  # [H, W, 1]
    else:
        raise ValueError("img_tensor shape must be [C,H,W], [H,W,C] or [H,W]")
    
    # 3. 归一化到0~255
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = img * 0  # 全0
    img = (img * 255).clamp(0, 255).to(torch.uint8)
    
    # 4. 转为numpy
    img_np = img.numpy()
    
    
    # 6. 保存
    im = Image.fromarray(img_np)
    im.save(filename)
    print(f"Saved to {filename}")



benchmark_map = {
    "libero_10": "LIBERO_10",
    "libero_spatial": "LIBERO_SPATIAL",
    "libero_object": "LIBERO_OBJECT",
    "libero_goal": "LIBERO_GOAL",
}

class ModelWrapper:
    def __init__(self, model, tokenizer, image_processor, cast_dtype, history_len=7, 
                use_ensembling=False, ensembling_temp=0.01, libero_eval_max_steps=600, 
                action_pred_steps=3, gripper_width=False):
        super().__init__()
        self.model = model
        self.cast_type = cast_dtype
        self.tokenizer = tokenizer
        self.image_process_fn = functools.partial(preprocess_image_vita, image_processor=image_processor)
        self.action_hist_queue = []
        self.history_len = history_len
        self.libero_eval_max_steps = libero_eval_max_steps
        self.action_pred_steps = action_pred_steps
        self.device = "cuda"
        self.use_ensembling = use_ensembling
        self.ensembling_temp = ensembling_temp
        self.img_queue = deque(maxlen=history_len)
        self.gripper_queue = deque(maxlen=history_len)
        self.state_queue = deque(maxlen=history_len)
        self.mask_queue = deque(maxlen=history_len)
        self.text_queue = deque(maxlen=history_len)
        self.act_queue = deque(maxlen=history_len-1)
        self.cnt = 0
        self.gripper_width = gripper_width
        if self.use_ensembling:
            self.all_time_actions = torch.zeros(
                    [
                        self.libero_eval_max_steps,
                        self.libero_eval_max_steps + self.action_pred_steps,
                        7,
                    ]
                ).to(self.device)

    def reset(self):
        self.img_queue = deque(maxlen=self.history_len)
        self.gripper_queue = deque(maxlen=self.history_len)
        self.state_queue = deque(maxlen=self.history_len)
        self.mask_queue = deque(maxlen=self.history_len)
        self.text_queue = deque(maxlen=self.history_len)
        self.act_queue = deque(maxlen=self.history_len-1)
        
        self.gripper_state = np.array([-1.0])
        if self.use_ensembling:
            self.all_time_actions = torch.zeros(
                    [
                        self.libero_eval_max_steps,
                        self.libero_eval_max_steps + self.action_pred_steps,
                        7,
                    ]
                ).to(self.device)

        self.cnt += 1

    def step(self, obs, goal, timestep):
        # preprocess image
        image = obs["agentview_image"]
        image = Image.fromarray(image)
        # image = image.transpose(Image.ROTATE_180)
        image_x = self.image_process_fn([image])
        # expand image dimension
        image_x = image_x.unsqueeze(1).to(dtype=self.cast_type)
        # save_tensor_as_png(image_x[0][0], f'./utils/images/test_imagex_{increment()}.png')

        gripper = obs["robot0_eye_in_hand_image"]
        gripper = Image.fromarray(gripper)
        gripper = self.image_process_fn([gripper])
        # expand image dimension
        gripper = gripper.unsqueeze(1).to(dtype=self.cast_type) 

        # expand text dimension
        text_x = goal
        
        state_pos = obs["robot0_eef_pos"]
        state_ori = quaternion_to_euler(obs["robot0_eef_quat"])

        if not self.gripper_width:
            state = torch.from_numpy(
                np.concatenate([state_pos, state_ori, self.gripper_state])).
            to(dtype=self.cast_type).unsqueeze(0).unsqueeze(0)  # [1, 1, 7]
        else:
            state = torch.from_numpy(
                np.concatenate([state_pos, state_ori, obs['robot0_gripper_qpos']])).
            to(dtype=self.cast_type).unsqueeze(0).unsqueeze(0)  # [1, 1, 8]

        with torch.no_grad():
            device = 'cuda'
            image_x = image_x.to(device) 
            gripper = gripper.to(device)
            state = state.to(device)

            self.img_queue.append(image_x)  
            self.gripper_queue.append(gripper)
            self.state_queue.append(state) 
            
            image_primary = torch.cat(list(self.img_queue), dim=1)
            image_wrist = torch.cat(list(self.gripper_queue), dim=1)
            state = torch.cat(list(self.state_queue), dim=1)
            
            
            batch_vita = prepare_data_vita(image_primary, image_wrist, goal, state, self.tokenizer)
            
            
            arm_action, gripper_action = self.model.module(
                input_ids = batch_vita['input_ids'],
                images = batch_vita['images'],
                attention_mask = batch_vita["attention_mask"],
                labels = batch_vita["labels"],
                states = batch_vita['states'],
                generate = True
            ) 
 
            
            arm_action = arm_action[:, -1]
            gripper_action = gripper_action[:, -1]
            # This need to align libero environment 
            if self.use_ensembling:
                # if num_step < self.history_len:
                #     selected_step = num_step - 1
                # else:
                #     selected_step = -1
                action = torch.concat((arm_action, gripper_action), dim=-1) # (1, action_pred_steps, 7)
                self.all_time_actions[timestep:timestep+1,timestep:timestep+self.action_pred_steps] = action 
                actions_for_curr_step = self.all_time_actions[:, timestep]
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]
                k = self.ensembling_temp
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = torch.from_numpy(exp_weights).to(self.device).unsqueeze(dim=1)
                action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                action = torch.concat((action[:, :6], action[:, 6:] > 0.5), dim=-1)
                action[:, -1] = (action[:, -1] - 0.5) * 2  # scale to -1 or 1
                action = action.detach().cpu().numpy()[-1]

        self.gripper_state = np.array([action[-1]])
        return action

from .visualize_eval_seq import generate_single_seq_gif, plot_and_save_gifs


def as_gif(sequences, path, lang):
    # Render the images as the gif (15Hz control frequency): 
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)    
    # if len(sequences) < 360:
    imgs = generate_single_seq_gif(sequences, lang)
    plot_and_save_gifs(imgs, path)
    print('video has been saved in {}'.format(path))

def collect_images(sequences, path, lang):
    # directory = os.path.dirname(path)
    os.makedirs(path, exist_ok=True)  
    for i in range(0, len(sequences), 5):
        img = Image.fromarray(sequences[i])
        img = img.rotate(180) 
        save_path = os.path.join(path, f'{i}.png')
        img.save(save_path)
        if i+5 >= len(sequences):
            img = Image.fromarray(sequences[len(sequences)-1])
            img = img.rotate(180) 
            save_path = os.path.join(path, f'{len(sequences)-1}.png')
            img.save(save_path)

def evaluate_libero_task(task, env, obs, args, model, cur_time='00'):
    steps = 0
    success = 0
    model.reset()
    goal = task.language
    sequences = []
    with torch.no_grad():
        while steps < args.libero_eval_max_steps: # default
            action = model.step(obs, goal, steps) 
            steps += 1
            
            obs, reward, done, info = env.step(action)
            sequences.append(obs["agentview_image"])
            
            
            if done:
                # collect_images(sequences=sequences, path=f"/videos_{cur_time}/{goal.replace(' ', '_')} \
                #                _{increment()}", lang=goal)
                
                success = 1
                break 
    env.close()
    return success

def evaluate_policy_ddp(args, model, cur_time='00'):
    pass 
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.finetune_type]()
    device_num = int(torch.distributed.get_world_size())
    device_id = torch.distributed.get_rank()
    results = []
    if "libero" in args.finetune_type:
        global num_eval_episodes 
        global task_num
        num_eval_episodes = 50
        task_num = 10
            
        NUM_SEQUENCES = num_eval_episodes * task_num 
        eval_sequences = list(range(NUM_SEQUENCES))
        assert NUM_SEQUENCES % device_num == 0
        interval_len = int(NUM_SEQUENCES // device_num)
        eval_sequences = eval_sequences[device_id*interval_len:min((device_id+1)*interval_len, NUM_SEQUENCES)]
        eval_sequences = tqdm(eval_sequences)
        
    else:
        raise NotImplementedError
    for eval_id in eval_sequences:
        task_id = eval_id // num_eval_episodes
        exp_id = eval_id % num_eval_episodes 
        task = task_suite.get_task(task_id)
        task_name = task.name
        task_description = task.language
        task_bddl_file = os.path.join(f"{args.libero_path}/libero/libero/bddl_files", 
                                      task.problem_folder, task.bddl_file)
        env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": args.libero_img_size,
        "camera_widths": args.libero_img_size,
        "render_gpu_device_id":device_id
        }
        print("device_id :", device_id)
        env = OffScreenRenderEnv(**env_args)
        env.task_id = task_id
        env.task_name = task_name
        env.task_suite_name = args.finetune_type
        env.reset()
        env.seed(args.seed)

        # set initial state
        init_states_path = os.path.join(
            f"{args.libero_path}/libero/libero/init_files", task.problem_folder, task.init_states_file
        )
        init_states = torch.load(init_states_path)
        init_state = init_states[exp_id]
        obs = env.set_init_state(init_state)

        for _ in range(5):  # simulate the physics without any actions
            env.step(np.zeros(7))

        result = evaluate_libero_task(task, env, obs, args, model, cur_time)
        results.append(result) 
        print("rank", torch.distributed.get_rank(), "results :", results)
    
    def merge_multi_list(res):
        tmp = []
        for l in res:
            tmp.extend(l)
        return tmp

    def extract_iter_from_tqdm(tqdm_iter):
        return [_ for _ in tqdm_iter]

    eval_sequences = extract_iter_from_tqdm(eval_sequences)
    res_tup = [(res, eval_seq) for res, eval_seq in zip(results, eval_sequences)]
    all_res_tup = [copy.deepcopy(res_tup) for _ in range(device_num)] if torch.distributed.get_rank() == 0 else None
    torch.distributed.gather_object(res_tup, all_res_tup, dst=0)

    if torch.distributed.get_rank() == 0:
        res_tup_list = merge_multi_list(all_res_tup)
        res_tup_list.sort(key=lambda x: x[1])
        print_and_save(res_tup_list, task_suite)

def print_and_save(result_list, task_suite):
    for j in range(task_num):
        this_result_list = result_list[j * num_eval_episodes: (j + 1) * num_eval_episodes]
        print("this_result_list :", this_result_list)
        this_result_list = np.array(this_result_list)
        avg_success = np.mean(this_result_list, axis=0)[0]
        task = task_suite.get_task(j)
        task_name = task.name
        print(f"Success rates for task {j} {task_name}:")
        print(f"{avg_success * 100:.1f}%")

def eval_one_epoch_libero_ddp(args, model, image_processor, tokenizer, cur_time):
    cast_dtype = get_cast_dtype(args.precision)
    hist_len = args.sequence_length
    wrapped_model = ModelWrapper(
                        model, 
                        tokenizer, 
                        image_processor, 
                        cast_dtype, 
                        history_len=hist_len, 
                        use_ensembling=args.eval_libero_ensembling,
                        ensembling_temp=args.ensembling_temp,
                        libero_eval_max_steps=args.libero_eval_max_steps,
                        action_pred_steps = args.action_pred_steps,
                        gripper_width=args.gripper_width)
    evaluate_policy_ddp(args, wrapped_model, cur_time)