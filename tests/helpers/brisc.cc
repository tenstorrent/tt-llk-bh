#include "tensix.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"

using namespace ckernel;

void device_setup() {

#ifdef ARCH_BLACKHOLE
    *((uint32_t volatile*)RISCV_DEBUG_REG_DEST_CG_CTRL) = 0;
#endif

    // WRITE_REG(RISCV_TDMA_REG_CLK_GATE_EN, 0x3f);  // Enable clock gating

    // set_deassert_addresses();

    // wzeromem(MEM_ZEROS_BASE, MEM_ZEROS_SIZE);

    // // Invalidate tensix icache for all 4 risc cores
    // cfg_regs[RISCV_IC_INVALIDATE_InvalidateAll_ADDR32] = RISCV_IC_BRISC_MASK | RISCV_IC_TRISC_ALL_MASK | RISCV_IC_NCRISC_MASK;
    
    // Clear destination registers
    TTI_ZEROACC(p_zeroacc::CLR_ALL, 0, 0, 1, 0);

    // Enable CC stack
	TTI_SFPENCC(3,0,0,10);
	TTI_NOP;

    // Set default sfpu constant register state
	TTI_SFPLOADI(p_sfpu::LREG0,0xA,0xbf80); // -1.0f -> LREG0
	TTI_SFPCONFIG(0, 11, 0); // LREG0 -> LREG11

//     // Enable ECC scrubber
//     core.ex_rmw_cfg(0, ECC_SCRUBBER_Enable_RMW, 1);
//     core.ex_rmw_cfg(0, ECC_SCRUBBER_Scrub_On_Error_RMW, 1);
//     core.ex_rmw_cfg(0, ECC_SCRUBBER_Delay_RMW, 0x100);

//     core.initialize_tensix_semaphores(instrn_buf[0]);

    // // unpacker semaphore
    // core.ex_sem_init(semaphore::UNPACK_MISC, 1, 1, instrn_buf[0]);

    // // unpacker sync semaphore
    // core.ex_sem_init(semaphore::UNPACK_SYNC, 2, 0, instrn_buf[0]);

    // // config state semaphore
    // core.ex_sem_init(semaphore::CFG_STATE_BUSY, MAX_CONFIG_STATES, 0, instrn_buf[0]);


}

int main(){
    device_setup();
}