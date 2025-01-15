import pytest
import torch
import os
from helpers import *

torch.set_printoptions(linewidth=500)

def generate_golden(operand1,format):
    return operand1

@pytest.mark.parametrize("format",  ["Int32"]) #["Bfp8_b", "Float16_b", "Float16", "Int32","Float32"])
@pytest.mark.parametrize("testname", ["eltwise_unary_datacopy_test"])
@pytest.mark.parametrize("dest_acc", ["DEST_ACC"])
def test_all(format, testname, dest_acc):
    #context = init_debuda()
    src_A,src_B = generate_stimuli(format)
    srcB = torch.full((1024,), 0)
    golden = generate_golden(src_A,format)
    write_stimuli_to_l1(src_A, src_B, format)

    make_cmd = f"make --silent format={format_args_dict[format]} testname={testname} dest_acc={dest_acc}"
    os.system(f"cd .. && {make_cmd}")

    run_elf_files(testname)

    if(format == "Float16" or format == "Float16_b"):
        read_words_cnt = len(src_A)//2
    elif( format == "Bfp8_b"):
        read_words_cnt = len(src_A)//4 + 32 # 272 for one tile
    elif( format == "Float32" or format == "Int32"):
        read_words_cnt = len(src_A)

    read_data = read_words_from_device("0,0", 0x1a000, word_count=read_words_cnt)
    
    read_data_bytes = flatten_list([int_to_bytes_list(data) for data in read_data])

    if(format == "Float16"):
        res_from_L1 = unpack_fp16(read_data_bytes)
    elif(format == "Float16_b"):
        res_from_L1 = unpack_bfp16(read_data_bytes)
    elif( format == "Bfp8_b"):
        res_from_L1 = unpack_bfp8_b(read_data_bytes)
    elif( format == "Float32"):
        res_from_L1 = unpack_float32(read_data_bytes)
    elif( format == "Int32"):
        res_from_L1 = unpack_int32(read_data_bytes)

    assert len(res_from_L1) == len(golden)

    os.system("cd .. && make clean")

    # Mailbox checks
    assert read_words_from_device("0,0", 0x19FF4, word_count=1)[0].to_bytes(4, 'big') == b'\x00\x00\x00\x01'
    assert read_words_from_device("0,0", 0x19FF8, word_count=1)[0].to_bytes(4, 'big') == b'\x00\x00\x00\x01'
    assert read_words_from_device("0,0", 0x19FFC, word_count=1)[0].to_bytes(4, 'big') == b'\x00\x00\x00\x01'

    if(format in format_dict):
        atol = 0.05
        rtol = 0.1
    else:
        atol = 0.2
        rtol = 0.1

    golden_tensor = torch.tensor(golden, dtype=format_dict[format] if format in ["Float16", "Float16_b", "Float32", "Int32"] else torch.bfloat16)
    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[format] if format in ["Float16", "Float16_b", "Float32", "Int32"] else torch.bfloat16)

    for i in range(len(golden)):
        assert torch.isclose(golden_tensor[i],res_tensor[i], rtol = rtol, atol = atol), f"Failed at index {i} with values {golden[i]} and {res_from_L1[i]}"
    
    _ , pcc = comp_pcc(golden_tensor, res_tensor, pcc=0.99) 
    assert pcc > 0.99
