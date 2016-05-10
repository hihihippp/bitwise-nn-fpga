#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "forward.h"

void writeToFile(unsigned int *x, unsigned int size)
{
  FILE *f = fopen("c.txt", "w");
  for (int i = 0; i < size; i++) {
    int v = x[i];
    for (int j = 0; j < 32; j++) {
      fprintf(f, "%d", v & 1);
      v <<= 1;
    }
    fprintf(f, "\n");
  }
  fclose(f);
}

void printImage(unsigned int *x)
{
	int count = 0;
	for (int i = 0; i < div32Ceil(28*28); i++) {
		unsigned int bits = x[i];
		for (int j = 0; j < 32; j++) {
			if (count % 28 == 0)
				printf("\n");
			if (bits & 0x80000000)
				printf("*");
			else
				printf(" ");
			bits <<= 1;
			count++;
		}
	}
	printf("\n");
}

int main () {
  FILE         *fp;

  int i, j;
  int space;

  unsigned int output;

  // read the weights
  FILE *fw = fopen("weights.bin", "rb");
  if (!fw) {
	  printf("Could not open weights file\n");
	  return 1;
  }

  space =  (div32Ceil(784)*2048 + 2048)*4 + 
           (div32Ceil(2048)*2048 + 2048)*4 + 
           (div32Ceil(2048)*2048 + 2048)*4 +
           (div32Ceil(2048)*10)*4;
  unsigned int *weights = (unsigned int *) malloc(space);
  fread(weights, 4, space/4, fw);

  // read the input
  FILE *fi = fopen("mnist.bin", "rb");
  if (!fi) {
	  printf("Could not open input file\n");
	  return 1;
  }
  int numExamples, width, height;
  fread(&numExamples, 4, 1, fi);
  fread(&height, 4, 1, fi);
  fread(&width, 4, 1, fi);
  int stride = div32Ceil(height * width);
  space = numExamples * stride * 4;
  unsigned int *examples = (unsigned int *) malloc(space);
  fread(examples, 4, space/4, fi);

  // read the labels
  FILE *fl = fopen("labels.bin", "rb");
  unsigned int *labels = (unsigned int *) malloc(4*numExamples);
  fread(labels, 4, numExamples, fl);

  numExamples = 100;
  double numCorrect = 0;
  for (i = 0; i < numExamples; i += 1) {

	// do a forward pass
    forward(&examples[i * stride], &output, weights);
    if (output == labels[i]) {
    	numCorrect++;
    }
    printf("Label: %d\n", labels[i]);
    printf("Output: %d\n\n", output);

  }

  double accuracy = 100*numCorrect/numExamples;
  printf("Accuracy: %f%%\n", accuracy);

  if(accuracy > 85) {
	  return 0;
  } else {
	  return 1;
  }

}
