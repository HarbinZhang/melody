#include <cufft.h>
#define NX 256
#define BATCH 10
cufftHandle plan;
cufftComplex *data;
cudaMalloc((void**)&data,sizeof(cufftComplex)*NX*BATCH);
/* Create a 1D FFT plan. */
cufftPlan1d(&plan,NX,CUFFT C2C,BATCH);
/* Use the CUFFT plan to transform the signal in place. */
cufftExecC2C(plan,data,data,CUFFT FORWARD);
/* Destroy the CUFFT plan. */
cufftDestroy(plan);
cudaFree(data);
