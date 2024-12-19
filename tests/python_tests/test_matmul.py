import pytest
import torch
import os
from ttlens.tt_lens_init import init_ttlens
from ttlens.tt_lens_lib import write_to_device, read_words_from_device, run_elf
from helpers import *

ELF_LOCATION = "../build/elf/"

def untilize(original_tensor, data_format):
    """
    Transform a 1D tensor of size 1024 (representing a 32x32 matrix in 16x16 submatrices)
    into a 32x32 tensor by interleaving the submatrices in row-major order.

    Args:
    - original_tensor (torch.Tensor): A 1D tensor of size 1024.

    Returns:
    - torch.Tensor: A 32x32 tensor with the interleaved submatrices.
    """
    # Ensure the input tensor is of size 1024
    if original_tensor.size(0) != 1024:
        raise ValueError("Input tensor must have 1024 elements.")

    submatrices = original_tensor.reshape(4, 16, 16)

    new_tensor = torch.zeros((32, 32), dtype=format_dict[data_format])

    for i in range(16):
        # Combine first 16 rows of submatrix 0 and submatrix 1
        new_tensor[i, :16] = submatrices[0, i, :]
        new_tensor[i, 16:] = submatrices[1, i, :]

        # Combine next 16 rows of submatrix 2 and submatrix 3
        new_tensor[i + 16, :16] = submatrices[2, i, :]
        new_tensor[i + 16, 16:] = submatrices[3, i, :]

    return new_tensor

def pearson_correlation(X, Y):
    """
    Compute the Pearson correlation coefficient between two torch tensors X and Y.
    
    Args:
    - X (torch.Tensor): First tensor.
    - Y (torch.Tensor): Second tensor.
    
    Returns:
    - torch.Tensor: Pearson correlation coefficient.
    """
    # Ensure the tensors are of the same size
    assert X.shape == Y.shape, "Tensors must have the same shape"
    
    # Calculate means
    mean_X = torch.mean(X)
    mean_Y = torch.mean(Y)
    
    # Compute the numerator and the denominator
    numerator = torch.sum((X - mean_X) * (Y - mean_Y))
    denominator = torch.sqrt(torch.sum((X - mean_X)**2) * torch.sum((Y - mean_Y)**2))
    
    # Return Pearson correlation coefficient
    return numerator / denominator

def run_elf_files(testname, run_brisc=True):

    if run_brisc == True:
        run_elf(f"{ELF_LOCATION}brisc.elf", "0,0", risc_id=0)

    for i in range(3):
        run_elf(f"{ELF_LOCATION}{testname}_trisc{i}.elf", "0,0", risc_id=i + 1)

def generate_golden(operand1, operand2, data_format):
    A_float = operand1.clone().detach().to(format_dict[data_format])
    B_float = operand2.clone().detach().to(format_dict[data_format])

    A_untilized = untilize(A_float,data_format)
    B_untilized = untilize(B_float,data_format)

    print(B_untilized)
    print(A_untilized)

    result = torch.matmul(B_untilized, A_untilized)
    print(result)
    result = result.view(-1)

    print(result[0])
    print(result[16])
    print(result[512])
    print(result[528])
    print("#"*50)

    return result.tolist()

def write_stimuli_to_l1(buffer_A, buffer_B, stimuli_format):
    if stimuli_format == "Float16_b":
        write_to_device("0,0", 0x1b000, pack_bfp16(buffer_A))
        write_to_device("0,0", 0x1c000, pack_bfp16(buffer_B))    
    elif stimuli_format == "Float16":
        write_to_device("0,0", 0x1b000, pack_fp16(buffer_A))
        write_to_device("0,0", 0x1c000, pack_fp16(buffer_B))

@pytest.mark.parametrize("format", ["Float16_b"])#, "Float16"])
@pytest.mark.parametrize("testname", ["matmul_test"])
def test_all(format, testname):

    #context = init_debuda()
    src_A, src_B = generate_stimuli(format)
    golden = generate_golden(src_A, src_B, format)
    write_stimuli_to_l1(src_A, src_B, format)

    make_cmd = f"make format={format_args_dict[format]} testname={testname}"
    os.system(f"cd .. && {make_cmd}")

    run_elf_files(testname)

    read_words_cnt = len(src_A) // (2 if format in ["Float16", "Float16_b"] else 1)
    read_data = read_words_from_device("0,0", 0x1a000, word_count=read_words_cnt)
    
    read_data_bytes = flatten_list([int_to_bytes_list(data) for data in read_data])
    
    res_from_L1 = unpack_bfp16(read_data_bytes) if format == "Float16_b" else unpack_fp16(read_data_bytes)

    assert len(res_from_L1) == len(golden)

    os.system("cd .. && make clean")

    # Mailbox checks
    assert read_words_from_device("0,0", 0x19FF4, word_count=1)[0].to_bytes(4, 'big') == b'\x00\x00\x00\x01'
    assert read_words_from_device("0,0", 0x19FF8, word_count=1)[0].to_bytes(4, 'big') == b'\x00\x00\x00\x01'
    assert read_words_from_device("0,0", 0x19FFC, word_count=1)[0].to_bytes(4, 'big') == b'\x00\x00\x00\x01'

    golden_tensor = torch.tensor(golden, dtype=format_dict[format] if format in ["Float16", "Float16_b"] else torch.bfloat16)
    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[format] if format in ["Float16", "Float16_b"] else torch.bfloat16)
    res_tensor = untilize(res_tensor,format).view(-1)

    if(format == "Float16_b" or format == "Float16"):
        atol = 0.2
        rtol = 0.1
    elif(format == "Bfp8_b"):
        atol = 0.4
        rtol = 0.3

    rel_errs = []

    #for i in range(len(golden)):
        #rel_errs.append(relative_error(golden_tensor[i], res_tensor[i]))
        #assert torch.isclose(golden_tensor[i],res_tensor[i], rtol = rtol, atol = atol), f"Failed at index {i} with values {golden[i]} and {res_from_L1[i]}"

    #print(rel_errs)
    print("*"*50)
    print(res_tensor[0])
    print(res_tensor[16])
    print(res_tensor[512])
    print(res_tensor[528])

    print("*"*50)
    print(golden_tensor[0])
    print(golden_tensor[16])
    print(golden_tensor[512])
    print(golden_tensor[528])
    print("#"*50)
    print("PCC")
    print(pearson_correlation(golden_tensor,res_tensor).item())
    assert 1==0