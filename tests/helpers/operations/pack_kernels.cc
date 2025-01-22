#include "pack_kernels.h"
#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

    #ifdef DEST_ACC
    const bool is_fp32_dest_acc_en = true;
    #else
    const bool is_fp32_dest_acc_en = false;
    #endif

    volatile uint32_t* buffer_Dest[PACK_ADDR_CNT] = {(volatile uint32_t*)0x1c000};

    inline void pack_init(){
        _llk_pack_hw_configure_<false, is_fp32_dest_acc_en, false>(DATA_FORMAT, DATA_FORMAT, 16*16*4);
        _llk_pack_init_<false, false, DstTileFaceLayout::RowMajor, false>(DATA_FORMAT);
        #ifdef ARCH_BLACKHOLE
        _llk_pack_dest_init_<DstSync::SyncFull,DstTileFaceLayout::RowMajor,is_fp32_dest_acc_en>();
        #else
        _llk_pack_dest_init_<DstSync::SyncFull, DstTileFaceLayout::RowMajor, false, is_fp32_dest_acc_en>();
        #endif
    }

    void pack_Dest_kernel(int index){ 
        // if(index == 0){   
        //     pack_init();
        // }
        pack_init();
        _llk_packer_wait_for_math_done_();
        _llk_pack_<DstSync::SyncFull,false, is_fp32_dest_acc_en>(0, (std::uint32_t)(buffer_Dest[index])/16-1);
        (*((volatile uint32_t*)0x17ff0 + index)) = (std::uint32_t)buffer_Dest[index];
        _llk_pack_dest_section_done_<DstSync::SyncFull,is_fp32_dest_acc_en>();
    }
