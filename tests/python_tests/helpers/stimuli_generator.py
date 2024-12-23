import torch
from .dictionaries import *

def flatten_list(sublists):
    return [item for sublist in sublists for item in sublist]

def generate_random_face(stimuli_format = "Float16_b"):

    if(stimuli_format == "Float16" or stimuli_format == "Float16_b"): 
        srcA_face = torch.ones(256, dtype = format_dict[stimuli_format]) + 2
    elif(stimuli_format == "Bfp8_b"):
        size = 256
        integer_part = torch.randint(0, 1, (size,))  
        fraction = torch.randint(0, 16, (size,)) / 16.0
        srcA_face = integer_part.float() + fraction 

    return srcA_face

def generate_random_face_ab(stimuli_format = "Float16_b"):
    return generate_random_face(), generate_random_face()

def generate_stimuli(stimuli_format = "Float16_b", tile_cnt = 1, sfpu = False):

    srcA = []
    srcB = []

    for i in range(4*tile_cnt):
        face_a, face_b = generate_random_face_ab(stimuli_format)
        srcA.append(face_a.tolist())
        srcB.append(face_b.tolist())

    srcA = flatten_list(srcA)
    srcB = flatten_list(srcB)

    if sfpu == False:
        if stimuli_format != "Bfp8_b":
            return torch.tensor(srcA, dtype = format_dict[stimuli_format]), torch.tensor(srcB, dtype = format_dict[stimuli_format])
        else:
            return torch.tensor(srcA, dtype = torch.bfloat16), torch.tensor(srcB, dtype = torch.bfloat16)
    else:
        srcA = generate_random_face(stimuli_format)
        srcB = torch.full((256,), 0, dtype = format_dict[stimuli_format])
        return srcA,srcB
