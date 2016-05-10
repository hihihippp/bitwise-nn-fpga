#include <stdio.h>
#include "forward.h"

#define SCRATCH_SIZE (4096/32)

inline u32 countBits(u32 x)
{
	u32 count = 0;
	countBits_label1:for (int i = 0; i < 32; i++) {
		count += x & 1;
		x >>= 1;
	}
	return count;
}

void forward (
	u32 input[INPUT_SIZE],
	u32 *output,
	u32 weights[WEIGHT_SIZE]
	) {

	static u32 scratch[2][SCRATCH_SIZE];

	u32 i, j, k;
	u32 acc;
	u32 max = 0;
	u32 class;
	u32 tmp = 0;

	// stream in the input
	InputLoop: for (i = 0; i < div32Ceil(28*28); i++) {
		scratch[0][i] = input[i];
	}

	// layer 1
	Layer1_Output_Loop: for (j = 0; j < 2048; j++) {
		acc = 0;
		Layer1_Input_Loop: for (k = 0; k < div32Ceil(784); k++) {
			u32 weight = weights[j*(div32Ceil(784)+1) + k];
			u32 x = scratch[0][k];
			acc += countBits(~(x ^ weight));
		}

		int mean = weights[j*(div32Ceil(784)+1) + k];
		acc -= mean;
		u32 out = (acc > (784 >> 1)) ? 1 : 0;
		u8 shift = j % 32;
		//u32 mask = 0x1 << shift;
		//scratch[1][j/32] = (scratch[1][j/32] & ~mask) | (out << shift);
		#if 1
		tmp = tmp | (out << shift);
		if (j % 32 == 31) {
			scratch[1][j/32] = tmp;
			tmp = 0;
		}
		#endif
	}

	// layer 2
	Layer2_Output_Loop: for (j = 0; j < 2048; j++) {
		acc = 0;
		Layer2_Input_Loop: for (k = 0; k < div32Ceil(2048); k++) {
			u32 weight = weights[LAYER1_SIZE + j*(div32Ceil(2048)+1) + k];
			u32 x = scratch[1][k];
			acc += countBits(~(x ^ weight));
		}

		int mean = weights[LAYER1_SIZE + j*(div32Ceil(2048)+1) + k];
		acc -= mean;
		u32 out = (acc > (2048 >> 1)) ? 1 : 0;
		u8 shift = j % 32;
		//u32 mask = 0x1 << shift;
		//scratch[0][j/32] = (scratch[0][j/32] & ~mask) | (out << shift);
		#if 1
		tmp = tmp | (out << shift);
		if (j % 32 == 31) {
			scratch[0][j/32] = tmp;
			tmp = 0;
		}
		#endif
	}

	// layer 3
	Layer3_Output_Loop: for (j = 0; j < 2048; j++) {
		acc = 0;
		Layer3_Input_Loop: for (k = 0; k < div32Ceil(2048); k++) {
			u32 weight = weights[LAYER1_SIZE+LAYER2_SIZE + j*(div32Ceil(2048)+1) + k];
			u32 x = scratch[0][k];
			acc += countBits(~(x ^ weight));
		}

		int mean = weights[LAYER1_SIZE+LAYER2_SIZE + j*(div32Ceil(2048)+1) + k];
		acc -= mean;
		u32 out = (acc > (2048 >> 1)) ? 1 : 0;
		u8 shift = j % 32;
		//u32 mask = 0x1 << shift;
		//scratch[1][j/32] = (scratch[1][j/32] & ~mask) | (out << shift);
		#if 1
		tmp = tmp | (out << shift);
		if (j % 32 == 31) {
			scratch[1][j/32] = tmp;
			tmp = 0;
		}
		#endif
	}

	// layer 4
	Layer4_Output_Loop: for (j = 0; j < 10; j++) {
		acc = 0;
		Layer4_Input_Loop: for (k = 0; k < div32Ceil(2048); k++) {
			u32 weight = weights[LAYER1_SIZE+LAYER2_SIZE+LAYER3_SIZE + j*div32Ceil(2048) + k];
			u32 x = scratch[1][k];
			acc += countBits(~(x ^ weight));
		}

		if (acc > max) {
			max = acc;
			class = j+1;
		}
	}
	*output = class;
}
