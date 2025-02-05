import pytest
import torch
import os
from helpers import *

test_results = []

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
formats = ["Bfp8_b", "Float16_b"]
#formats = ["Float16_b", "Float16_b"]
#try_formats = ["Bfp8_b", "Bfp8_b"]
mathops = ["elwadd", "elwsub", "elwmul"]
mathops = ["elwadd"]
accumulation = ["", "DEST_ACC"]
accumulation = [""]
# @pytest.mark.parametrize("input_format", formats)
# @pytest.mark.parametrize("output_format", formats)
@pytest.mark.parametrize("unpack_src", formats)
@pytest.mark.parametrize("unpack_dst", formats)
#@pytest.mark.parametrize("math_src", formats)
@pytest.mark.parametrize("math_dst", formats)
@pytest.mark.parametrize("pack_src", formats)
@pytest.mark.parametrize("pack_dst", formats)
@pytest.mark.parametrize("testname", ["eltwise_binary_test"])
@pytest.mark.parametrize("mathop", mathops)
@pytest.mark.parametrize("dest_acc", accumulation)
def test_all(unpack_src, unpack_dst, math_dst, pack_src, pack_dst, mathop, testname, dest_acc):
    #context = init_debuda()
    # if unpack_dst != math_dst and math_dst != pack_src:
    #     pytest.skip("")
    src_A, src_B = generate_stimuli(unpack_src)
    print("SRC_A = ", src_A)
    print("SRC_B = ", src_B)
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
    print()
    print("GOLDEN = ", golden)
    
    test_results.append([testname, mathop, unpack_src,unpack_dst, math_dst, pack_src, pack_dst,"FAIL"])
    
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
    print("RESULT TENSOR = ", res_tensor)
    print("GOLDEN TENSOR= ", golden_tensor)

    
    for i in range(len(golden)):
        assert torch.isclose(golden_tensor[i],res_tensor[i], rtol = rtol, atol = atol), f"Failed at index {i} with values {golden[i]} and {res_from_L1[i]}"

    _ , pcc = comp_pcc(golden_tensor, res_tensor, pcc=0.99) 
    assert pcc > 0.99
    test_results[-1][-1] = "PASS"


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    # Print header
    print(test_results)
    # terminalreporter.write_sep("=", "Test Results Summary")
    
    # # Print table header
    # terminalreporter.write(f"{'TEST':<10}{'NAME':<10}{'UNPACK SRC':<10}{'UNPACK DST':<10}{'FPU':<10}{'PACK SRC':<10}{'PACK DST':<10}{'RESULT':<10}\n")
    
    # # Print results of each test
    # for result in test_results:
    #     testname, mathop, unpack_src, unpack_dst, math_dst, pack_src, pack_dst, outcome = result
    #     terminalreporter.write(f"{testname:<10}{mathop:<10}{unpack_src:<10}{unpack_dst:<10}{math_dst:<10}{pack_src:<10}{pack_dst:<10}{outcome:<10}\n")

    # # Optional: Check the overall result (if you want to customize it further)
    # if exitstatus == 0:
    #     terminalreporter.write("\nAll tests passed.\n")
    # else:
    #     terminalreporter.write("\nSome tests failed.\n")