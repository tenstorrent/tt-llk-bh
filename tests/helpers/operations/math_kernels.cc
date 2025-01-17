#include "math_kernels.h"
#include "params.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h"

    #ifdef DEST_ACC
    const bool is_fp32_dest_acc_en = true;
    #else
    const bool is_fp32_dest_acc_en = false;
    #endif

    inline void math_init(int i){
        _llk_math_pack_sync_init_<DstSync::SyncFull,is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<false,false>(DATA_FORMAT,DATA_FORMAT);
        if(kernels[i] == &elwadd_kernel){
            _llk_math_eltwise_binary_init_<EltwiseBinaryType::ELWADD, BroadcastType::NONE>(4, 0, 0);
        }else if(kernels[i] == &elwsub_kernel){
            _llk_math_eltwise_binary_init_<EltwiseBinaryType::ELWSUB, BroadcastType::NONE>(4, 0, 0);
        }else if(kernels[i] == &elwmul_kernel){
            _llk_math_eltwise_binary_init_<EltwiseBinaryType::ELWMUL, BroadcastType::NONE>(4, 0, 0);
        }
    }

    void elwadd_kernel(int i){
        if(i==0 || (i!=0 && kernels[i] != kernels[i-1])){
            math_init(i);
        }
        _llk_math_wait_for_dest_available_<DstSync::SyncFull>();
        _llk_math_eltwise_binary_<EltwiseBinaryType::ELWADD, BroadcastType::NONE,DstSync::SyncFull, 0, EltwiseBinaryReuseDestType::NONE, is_fp32_dest_acc_en>(4, 0, false);
        _llk_math_dest_section_done_<DstSync::SyncFull,is_fp32_dest_acc_en>();
    }

    void elwsub_kernel(int i){
        if(i==0 || (i!=0 && kernels[i] != kernels[i-1])){
            math_init(i);
        }
        _llk_math_wait_for_dest_available_<DstSync::SyncFull>();
        _llk_math_eltwise_binary_<EltwiseBinaryType::ELWSUB, BroadcastType::NONE,DstSync::SyncFull, 0, EltwiseBinaryReuseDestType::NONE, is_fp32_dest_acc_en>(4, 0, false);
        _llk_math_dest_section_done_<DstSync::SyncFull,is_fp32_dest_acc_en>();
    }

    void elwmul_kernel(int i){
        if(i==0 || (i!=0 && kernels[i] != kernels[i-1])){
            math_init(i);
        }
        _llk_math_wait_for_dest_available_<DstSync::SyncFull>();
        _llk_math_eltwise_binary_<EltwiseBinaryType::ELWMUL, BroadcastType::NONE,DstSync::SyncFull, 0, EltwiseBinaryReuseDestType::NONE, is_fp32_dest_acc_en>(4, 0, false);
        _llk_math_dest_section_done_<DstSync::SyncFull,is_fp32_dest_acc_en>();
    }

    //TODO: ADD MORE