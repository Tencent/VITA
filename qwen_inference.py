from transformers import AutoModelForCausalLM, AutoTokenizer

#model_name = "/mnt/cfs/lhj/model_weights/Qwen2.5-7B-Instruct"
#model_name = "/mnt/cfs2/lhj/videomllm_ckpt/outputs/vita_video_audio_0924/Qwen2.5-7B-Instruct-vita0924ckpt9000"
model_name = "/mnt/cfs2/lhj/videomllm_ckpt/outputs/vita_video_audio_1021/Qwen2.5-7B-Instruct-vita1021s3_neg"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "你会中文吗？此去泉台招旧部，下一句是是什么？"
prompt = "小明的爸爸有3个儿子，大儿子叫大毛，二儿子叫二毛，请问三儿子叫啥？"
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt},
#    {"role": "assistant", "content": "小明。"},
#    {"role": "user", "content": "你是谁？"},
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
import pdb; pdb.set_trace()
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

#import pdb; pdb.set_trace()
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
print('-------------------------------------------------------------')
print(tokenizer.decode(model_inputs['input_ids'][0]))
print('-------------------------------------------------------------')
print(tokenizer.decode(generated_ids[0]))
print('-------------------------------------------------------------')
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
