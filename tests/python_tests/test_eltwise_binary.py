import pytest
import torch
import os
from helpers import *

ELF_LOCATION = "../build/elf/"

def generate_golden(operation, operand1, operand2, data_format):
    if( data_format == "Float16" or data_format == "Float16_b"):
        tensor1_float = operand1.clone().detach().to(format_dict[data_format])
        tensor2_float = operand2.clone().detach().to(format_dict[data_format])
    else:
        tensor1_float = operand1.clone().detach().to(format_dict["Float16_b"])
        tensor2_float = operand2.clone().detach().to(format_dict["Float16_b"])
    
    operations = {
        "elwadd": tensor1_float + tensor2_float,
        "elwsub": tensor1_float - tensor2_float,
        "elwmul": tensor1_float * tensor2_float
    }
    
    if operation not in operations:
        raise ValueError("Unsupported operation!")

    return operations[operation].tolist()

@pytest.mark.parametrize("format", ["Bfp8_b", "Float16_b", "Float16"])
@pytest.mark.parametrize("testname", ["eltwise_binary_test"])
@pytest.mark.parametrize("mathop", ["elwadd", "elwsub", "elwmul"])
def test_all(format, mathop, testname):
    #context = init_debuda()
    src_A, src_B = generate_stimuli(format)
    golden = generate_golden(mathop, src_A, src_B, format)
    write_stimuli_to_l1(src_A, src_B, format)

    make_cmd = f"make --silent format={format_args_dict[format]} mathop={mathop_args_dict[mathop]} testname={testname}"
    os.system(f"cd .. && {make_cmd}")

    run_elf_files(testname)

    if(format == "Float16" or format == "Float16_b"):
        read_words_cnt = len(src_A)//2
    elif( format == "Bfp8_b"):
        read_words_cnt = len(src_A)//4 + 64//4 # 272 for one tile

    read_data = read_words_from_device("0,0", 0x1a000, word_count=read_words_cnt)
    
    read_data_bytes = flatten_list([int_to_bytes_list(data) for data in read_data])

    if(format == "Float16"):
        res_from_L1 = unpack_fp16(read_data_bytes)
    elif(format == "Float16_b"):
        res_from_L1 = unpack_bfp16(read_data_bytes)
    elif( format == "Bfp8_b"):
        res_from_L1 = unpack_bfp8_b(read_data_bytes)

    assert len(res_from_L1) == len(golden)

    os.system("cd .. && make clean")

    # Mailbox checks
    assert read_words_from_device("0,0", 0x19FF4, word_count=1)[0].to_bytes(4, 'big') == b'\x00\x00\x00\x01'
    assert read_words_from_device("0,0", 0x19FF8, word_count=1)[0].to_bytes(4, 'big') == b'\x00\x00\x00\x01'
    assert read_words_from_device("0,0", 0x19FFC, word_count=1)[0].to_bytes(4, 'big') == b'\x00\x00\x00\x01'

    if(format == "Float16_b" or format == "Float16"):
        atol = 0.05
        rtol = 0.1
    elif(format == "Bfp8_b"):
        atol = 0.4
        rtol = 0.3

    golden_tensor = torch.tensor(golden, dtype=format_dict[format] if format in ["Float16", "Float16_b"] else torch.bfloat16)
    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[format] if format in ["Float16", "Float16_b"] else torch.bfloat16)

    for i in range(len(golden)):
        assert torch.isclose(golden_tensor[i],res_tensor[i], rtol = rtol, atol = atol), f"Failed at index {i} with values {golden[i]} and {res_from_L1[i]}"

    _ , pcc = comp_pcc(golden_tensor, res_tensor, pcc=0.99) 
    assert pcc > 0.99