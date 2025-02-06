import pytest
import torch
import os
from helpers import *

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
        "elwmul": tensor1_float * tensor2_float,
        "elwdiv": tensor1_float / tensor2_float
    }
    
    if operation not in operations:
        raise ValueError("Unsupported operation!")

    return operations[operation].tolist()

formats = ["Bfp8_b", "Float16_b", "Float16"]
mathops = ["elwadd", "elwsub", "elwmul"]
accumulation = ["", "DEST_ACC"]
@pytest.mark.parametrize("unpack_src", formats)
@pytest.mark.parametrize("unpack_dst", formats)
@pytest.mark.parametrize("math_dst", formats)
@pytest.mark.parametrize("pack_src", formats)
@pytest.mark.parametrize("pack_dst", formats)
@pytest.mark.parametrize("testname", ["eltwise_binary_test"])
@pytest.mark.parametrize("mathop", mathops)
@pytest.mark.parametrize("dest_acc", accumulation)
def test_all(unpack_src, unpack_dst, math_dst, pack_src, pack_dst, mathop, testname, dest_acc, test_results):
    #context = init_debuda()

    src_A, src_B = generate_stimuli(unpack_src)
    golden = generate_golden(mathop, src_A, src_B, pack_dst)
    write_stimuli_to_l1(src_A, src_B, unpack_src)

    test_config = {
        "unpack_src": unpack_src,
        "unpack_dst": unpack_dst,
        "math_dst": math_dst,
        "pack_src": pack_src,
        "pack_dst": pack_dst,
        "testname": testname,
        "dest_acc": dest_acc,
        "mathop": mathop
    }

    make_cmd = generate_make_command(test_config)
    os.system(f"cd .. && {make_cmd}")

    run_elf_files(testname)
    # os.system("cd .. && make clean")
    # os.system("tt-smi -r 0")
    
    res_from_L1 = collect_results(pack_dst,src_A)
    
    acc = "OFF"
    if dest_acc != "":
        acc = "ON"
    test_results.append(["FAIL", unpack_src, pack_dst, 0, "deadbeef",unpack_src,unpack_dst, math_dst, pack_src, pack_dst, mathop, acc])
    # test_results.append(["FAIL", unpack_src, pack_dst, unpack_src,unpack_dst, math_dst, pack_src, pack_dst, mathop, acc])
    
    assert len(res_from_L1) == len(golden)

    os.system("cd .. && make clean")
    os.system("tt-smi -r 0")

    # Mailbox checks
    assert read_words_from_device("0,0", 0x19FF4, word_count=1)[0].to_bytes(4, 'big') == b'\x00\x00\x00\x01'
    assert read_words_from_device("0,0", 0x19FF8, word_count=1)[0].to_bytes(4, 'big') == b'\x00\x00\x00\x01'
    assert read_words_from_device("0,0", 0x19FFC, word_count=1)[0].to_bytes(4, 'big') == b'\x00\x00\x00\x01'

    if(pack_dst == "Float16_b" or pack_dst == "Float16"):
        atol = 0.05
        rtol = 0.1
    elif(pack_dst == "Bfp8_b"):
        atol = 0.1
        rtol = 0.2

    golden_tensor = torch.tensor(golden, dtype=format_dict[pack_dst] if pack_dst in ["Float16", "Float16_b"] else torch.bfloat16)
    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[pack_dst] if pack_dst in ["Float16", "Float16_b"] else torch.bfloat16)
    
    for i in range(len(golden)):
        if str(res_tensor[i]) not in [str(-111.0), str(-6.232981884280766e+18), str(-427.25)]:
            test_results[-1][4] = res_tensor[i]
        assert torch.isclose(golden_tensor[i],res_tensor[i], rtol = rtol, atol = atol), f"Failed at index {i} with values {golden[i]} and {res_from_L1[i]}"

    _ , pcc = comp_pcc(golden_tensor, res_tensor, pcc=0.99) 
    assert pcc > 0.99
    test_results[-1][3] = pcc
    test_results[-1][0] = "PASS"