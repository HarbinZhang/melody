//includes,project
#include<cuda_runtime.h>
#include<cufft.h>
#include<iostream>
using namespace std;
#define SIGNAL_SIZE 16
typedef unsigned int uint32;
typedef unsigned long int uint64;

int main()
{
        //Allocate host memory for the signal
        cufftComplex* h_signal=(cufftComplex *)malloc(sizeof(cufftComplex) *SIGNAL_SIZE);

        //Initialize the memory for the signal
        for(unsigned int i=0;i<SIGNAL_SIZE;++i)
        {
                h_signal[i].x=1;
                h_signal[i].y=0;
        }
        
        //Allocate device memory for signal
        cufftComplex *d_signal;
        cudaMalloc((void **)&d_signal,sizeof(cufftComplex)*SIGNAL_SIZE);

        //Copy host memory to device
        cudaMemcpy(d_signal,h_signal,sizeof(cufftComplex)*SIGNAL_SIZE,cudaMemcpyHostToDevice);

        //CUFFT plan
        cufftHandle plan;
        // cufftPlan1d(&plan,SIGNAL_SIZE,CUFFT_C2C,1);

        int n[1] = {3};
        cufftResult res = cufftPlanMany(&plan, 1, n,
        NULL, 1, 0,  //advanced data layout, NULL shuts it off
        NULL, 1, 0,  //advanced data layout, NULL shuts it off
        CUFFT_C2C, 4);    

        
        //Transform signal 
        cufftExecC2C(plan,(cufftComplex *)d_signal,(cufftComplex *)d_signal,CUFFT_FORWARD);

        //Copy device memory to host
        cudaMemcpy(h_signal,d_signal,sizeof(cufftComplex)*SIGNAL_SIZE,cudaMemcpyDeviceToHost);

        for(unsigned int i=0;i<SIGNAL_SIZE;++i)
        {
                // cout<<h_signal[i].x<<endl;
                // cout<<h_signal[i].y<<endl;
                printf("%f\n", h_signal[i].x);
        }
        
        //Destory CUFFT context
        cufftDestroy(plan);
        
        //cleanup memory
        free(h_signal);
        cudaFree(d_signal);
        
        cudaDeviceReset();
}