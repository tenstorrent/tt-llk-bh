import pytest
import torch
import os
from helpers import *

# ANSI escape codes for colors
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
RESET = "\033[0m"  # Reset to default color

def modify_list(input_list):
    # First 64 elements stay the same
    result = input_list[:64]
    
    # For the remaining elements, group every 4 and reverse the groups
    for i in range(64, len(input_list), 4):
        # Get the next 4 elements and reverse the order
        group = input_list[i:i+4][::-1]
        result.extend(group)
    
    return result

torch.set_printoptions(linewidth=500)

def generate_golden(operand1,format):
    return operand1, pack_bfp8_b(operand1)

@pytest.mark.parametrize("format", ["Bfp8_b"])#Float16_b", "Float16"]) #,"Float32", "Int32"])
@pytest.mark.parametrize("testname", ["eltwise_unary_datacopy_test"])
@pytest.mark.parametrize("dest_acc", [""])#,"DEST_ACC"])
def test_all(format, testname, dest_acc):
    #context = init_debuda()
    src_A,src_B = generate_stimuli(format)
    srcB = torch.full((1024,), 0)
    golden,packed_garbage = generate_golden(src_A,format)
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

    for i in range(len(packed_garbage)):
        if(i%4 ==0 and i!=0):
            print("\n\n")
        print(f"{i}: {packed_garbage[i]} | {read_data_bytes[i]} {YELLOW}{packed_garbage[i]==read_data_bytes[i]}{RESET}")

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

    if(format == "Float16_b" or format == "Float16" or format == "Float32"):
        atol = 0.05
        rtol = 0.1
    elif(format == "Bfp8_b"):
        atol = 0.4
        rtol = 0.3

    golden_tensor = torch.tensor(golden, dtype=format_dict[format] if format in ["Float16", "Float16_b", "Float32", "Int32"] else torch.bfloat16)
    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[format] if format in ["Float16", "Float16_b", "Float32", "Int32"] else torch.bfloat16)

    golden_str = ""
    res_str = ""
    failed_goldens = []
    failed_res = []

    for golden_val, res_val in zip(golden_tensor.flatten(), res_tensor.flatten()):
        if golden_val == res_val:
            # Print matching values in green (same value in both tensors)
            golden_str += f"{GREEN}{golden_val.item()} {RESET}"
            res_str += f"{GREEN}{res_val.item()} {RESET}"
        else:
            # Print differing values in red
            golden_str += f"{RED}{golden_val.item()} {RESET}"
            res_str += f"{RED}{res_val.item()} {RESET}"
            failed_goldens.append(golden_val.item())
            failed_res.append(res_val.item())


    # Print the entire row of tensors
    print(golden_str)
    print("\n"*5)
    print(res_str)
    print("\n"*5)
    # print(failed_res)
    # print("\n"*5)
    # print(failed_goldens)
    
    # print("#"*200)

    # for i in range(len(failed_goldens)):
    #     tensor1 = torch.full((1024,), failed_goldens[i])
    #     tensor2 = torch.full((1024,), failed_res[i])

    #     print(tensor1[0].item())
    #     print(pack_bfp8_b(tensor1))
    #     print("\n"*5)
    #     print(tensor2[0].item())
    #     print(pack_bfp8_b(tensor2))
    #     print("*"*100)

    for i in range(len(golden)):
        assert torch.isclose(golden_tensor[i],res_tensor[i], rtol = rtol, atol = atol), f"Failed at index {i} with values {golden[i]} and {res_from_L1[i]}"
    
    _ , pcc = comp_pcc(golden_tensor, res_tensor, pcc=0.99) 
    assert pcc > 0.99
