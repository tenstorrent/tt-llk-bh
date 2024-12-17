// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ckernel.h"
#include "ckernel_addr_map.h"
#include "ckernel_pcbuf.h"
#include "ckernel_main.h"
#include "ckernel_globals.h"
#include <l1_address_map.h>
#include <tensix.h>
// Necessary for ckernel variables
#include "ckernel_helper.h"

using vptr_uint = volatile uint32_t*;

#ifdef LLK_TRISC_UNPACK
	volatile uint32_t* mailbox = (volatile uint32_t*)(0x19FFC);
	#ifdef MULTIPLE_OPS
		#include "operations/unpack_kernels.h"
	#endif
#elif defined(LLK_TRISC_MATH)
	volatile uint32_t* mailbox = (volatile uint32_t*)(0x19FF8);
	#ifdef MULTIPLE_OPS
		#include "operations/math_kernels.h"
	#endif
#elif defined(LLK_TRISC_PACK)
	volatile uint32_t* mailbox = (volatile uint32_t*)(0x19FF4);
	#ifdef MULTIPLE_OPS
		#include "operations/pack_kernels.h"
	#endif
#endif


// inline void c_tensix_core::ex_load_const(vptr_uint instrn_buf) {
//     // Load LREG11 w/ -1.0f by convention
//     uint instrn;
//     instrn = (0xbf80 << 0);  // Load LREG0 w/ -1.0f
//     ex_push_insn(instrn_buf, INSTRN_SFPLOADI(instrn));
//     instrn = (11 << 4);  // Set LREG11 to LREG0
//     ex_push_insn(instrn_buf, INSTRN_SFPCONFIG(instrn));
// }

//   TTI_SFPLOAD(p_sfpu::LREG3, 0, 0, 2); // Save dest addr 0 (odd cols)  to LREG_3

// core.ex_rmw_cfg(0, ECC_SCRUBBER_Enable_RMW, 1);
// core.ex_rmw_cfg(0, ECC_SCRUBBER_Scrub_On_Error_RMW, 1);
// core.ex_rmw_cfg(0, ECC_SCRUBBER_Delay_RMW, 0x100);

// inline uint32_t rmw_cfg_value(uint cfg_shamt, uint32_t cfg_mask, uint32_t wrdata, uint32_t l_cfg_data) {
//     uint32_t cfg_data = l_cfg_data;

//     // Shift and mask wrdata to properly align withn 32-bit DWORD
//     wrdata <<= cfg_shamt;
//     wrdata &= cfg_mask;

//     // Zero-out relevant bits in cfg data
//     cfg_data &= ~cfg_mask;

//     // Or new data bits
//     cfg_data |= wrdata;

//     return cfg_data;
// }

// inline void ex_rmw_cfg(uint cfg_addr32, uint cfg_shamt, uint32_t cfg_mask, uint wr_val, uint32_t cfg_regs) {
//     uint32_t addr = cfg_addr32;
//     uint32_t cfg_data = cfg_regs[addr];
// 	cfg_regs[addr] = rmw_cfg_value(cfg_shamt, cfg_mask, wr_val, cfg_data);
// }

int main()
{
    //FWEVENT("Launching proudction env kernels");
	for (int i = 0; i < 64; i++) regfile[i] = 0;
	reset_cfg_state_id();
	reset_dest_offset_id();

	#ifdef LLK_TRISC_MATH
    TTI_ZEROACC(p_zeroacc::CLR_ALL, 0, 0, 1, 0);
	TTI_SFPENCC(3,0,0,10);
	TTI_NOP;
	TTI_SFPLOADI(p_sfpu::LREG0,0xA,0xbf80); // -1.0f -> LREG0
	TTI_SFPCONFIG(0, 11, 0); // LREG0 -> LREG11
	#endif

	#ifdef MULTIPLE_OPS
	// needs these 2 defines when compiling
	PROCESS_NUMBERS(KERN_CNT, KERNS);
		#ifdef LLK_TRISC_PACK
			PROCESS_ADDRESSES(PACK_ADDR_CNT,PACK_ADDRS);
		#endif
	#endif

	tensix_sync();
    run_kernel();

	*mailbox = KERNEL_COMPLETE; // 0x1

	for(;;){}
}