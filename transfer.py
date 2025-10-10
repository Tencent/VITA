import torch

# Load the original checkpoint
checkpoint = torch.load('path/to/original/checkpoint/1.pth', map_location='cpu')

# Filter out all weights that contain 'vlm'
old_state_dict = checkpoint['model_state_dict']

new_state_dict = {k: v for k, v in old_state_dict.items() if 'vlm' not in k}

print(new_state_dict.keys())

# Construct a new checkpoint
new_checkpoint = {
    'epoch': checkpoint['epoch'],
    'model_state_dict': new_state_dict
}

def get_size(state_dict):
    """Calculate the size of the state dictionary in MB."""
    return sum(v.numel() * v.element_size() for v in state_dict.values()) / 1024 / 1024

print("Original size: %.2f MB" % get_size(old_state_dict))
print("New size: %.2f MB" % get_size(new_state_dict))

# Save to a new weight file
torch.save(new_checkpoint, 'path/to/save/new/checkpoint/vita_seer_717_ft_pt_small.pth')

# Load the new checkpoint to verify
ckpt = torch.load('path/to/save/new/checkpoint/vita_seer_717_ft_pt_small.pth', map_location='cpu')
print(ckpt.keys())
print(sum(v.numel() * v.element_size() for v in ckpt['model_state_dict'].values()) / 1024 / 1024, 'MB')
