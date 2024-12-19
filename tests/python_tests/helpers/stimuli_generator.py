import torch
from .dictionaries import *

def flatten_list(sublists):
    return [item for sublist in sublists for item in sublist]

def generate_one_random_face(stimuli_format):

    if(stimuli_format == "Float16" or stimuli_format == "Float16_b"): 
        # srcA_face = torch.rand(256, dtype = format_dict[stimuli_format]) + 2
        # srcB_face = torch.rand(256, dtype = format_dict[stimuli_format]) + 2

        srcA_face = torch.full((256,), torch.rand(1)[0].item(),dtype = format_dict[stimuli_format])
        srcB_face = torch.full((256,), torch.rand(1)[0].item(),dtype = format_dict[stimuli_format])

    elif(stimuli_format == "Bfp8_b"):
        size = 256
        integer_part = torch.randint(0, 1, (size,))  
        fraction = torch.randint(0, 16, (size,)) / 16.0
        srcA_face = integer_part.float() + fraction 
        integer_part = torch.randint(0, 1, (size,))  
        fraction = torch.randint(0, 16, (size,)) / 16.0
        srcB_face = integer_part.float() + fraction  

    return srcA_face, srcB_face

def generate_stimuli(stimuli_format, tile_cnt = 1):

    srcA = []
    srcB = []

    for i in range(4*tile_cnt):
        face_a, face_b = generate_one_random_face(stimuli_format)
        srcA.append(face_a.tolist())
        srcB.append(face_b.tolist())

    if stimuli_format != "Bfp8_b":
        return torch.tensor([item for sublist in srcA for item in sublist], dtype = format_dict[stimuli_format]), torch.tensor([item for sublist in srcB for item in sublist], dtype = format_dict[stimuli_format])
    else:
        return torch.tensor([item for sublist in srcA for item in sublist], dtype = torch.bfloat16), torch.tensor([item for sublist in srcB for item in sublist], dtype = torch.bfloat16)