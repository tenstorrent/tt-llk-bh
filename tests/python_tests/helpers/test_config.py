from .dictionaries import *

def generate_make_command(test_config):
    make_cmd = f"make --silent "

    unpack_src = test_config.get("unpack_src", "Float16_b") # Flolat16_b is default
    unpack_dst = test_config.get("unpack_dst", "Float16_b")
    math_dst = test_config.get("math_dst", "Float16_b")
    pack_src = test_config.get("pack_src", "Float16_b") # Flolat16_b is default
    pack_dst = test_config.get("pack_dst", "Float16_b")
    testname = test_config.get("testname")
    dest_acc = test_config.get("dest_acc", " ") # default is not 32 bit dest_acc 

    make_cmd += f"unpack_src={unpack_src_dict[unpack_src]} unpack_dst={unpack_dst_dict[unpack_dst]} math_dst={math_dst_dict[math_dst]} pack_src={pack_src_dict[pack_src]} pack_dst={pack_dst_dict[pack_dst]} " # jsut for now take output_format
    make_cmd += f"testname={testname} dest_acc={dest_acc} "
    mathop = test_config.get("mathop", "no_mathop")
    approx_mode = test_config.get("approx_mode","false")

    make_cmd += f" approx_mode={approx_mode} "

    if(mathop != "no_mathop"):
        if isinstance(mathop,str): # single tile option
            make_cmd += f"mathop={mathop_args_dict[mathop]}"
        else: # multiple tiles handles mathop as int

            if(mathop == 1):
                make_cmd += " mathop=ELTWISE_BINARY_ADD "
            elif(mathop == 2):
                make_cmd += " mathop=ELTWISE_BINARY_SUB "
            else:
                make_cmd += " mathop=ELTWISE_BINARY_MUL "

            kern_cnt = str(test_config.get("kern_cnt"))
            pack_addr_cnt = str(test_config.get("pack_addr_cnt"))
            pack_addrs = test_config.get("pack_addrs")
            unpack_a_addrs_cnt = test_config.get("unpack_a_addrs_cnt")

            make_cmd += f" kern_cnt={kern_cnt} "
            make_cmd += f" pack_addr_cnt={pack_addr_cnt} pack_addrs={pack_addrs}" 
            make_cmd += f" unpack_a_addrs_cnt={unpack_a_addrs_cnt}"



    return make_cmd