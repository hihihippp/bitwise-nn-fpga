#ifndef FORWARD_H_
#define FORWARD_H_

#define roundUp32(x) ( (((x-1)>>5) + 1) << 5 )
#define div32Ceil(x) (((x-1)>>5) + 1)

#define LAYER1_SIZE ( div32Ceil(784)*2048 + 2048 )
#define LAYER2_SIZE ( div32Ceil(2048)*2048 + 2048 )
#define LAYER3_SIZE ( div32Ceil(2048)*2048 + 2048 )
#define LAYER4_SIZE ( div32Ceil(2048)*10 )
#define WEIGHT_SIZE ( LAYER1_SIZE + LAYER2_SIZE + LAYER3_SIZE + LAYER4_SIZE )
#define INPUT_SIZE  ( div32Ceil(28*28) )

typedef unsigned char u8;
typedef unsigned int u32;

void forward(u32 input[INPUT_SIZE], u32 *output, u32 weights[WEIGHT_SIZE]);

#endif
