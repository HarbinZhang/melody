#include <stdio.h>
#include <math.h>
#include <ctime>
#include <chrono>
#include <cufft.h>

#define BUFFER_SIZE 4096

// cufftComplex data type
// typedef float2 cufftComplex;

#define SIGNAL_SIZE 4000

typedef struct  WAV_HEADER
{
    /* RIFF Chunk Descriptor */
    uint8_t         RIFF[4];        // RIFF Header Magic header
    uint32_t        ChunkSize;      // RIFF Chunk Size
    uint8_t         WAVE[4];        // WAVE Header
    /* "fmt" sub-chunk */
    uint8_t         fmt[4];         // FMT header
    uint32_t        Subchunk1Size;  // Size of the fmt chunk
    uint16_t        AudioFormat;    // Audio format 1=PCM,6=mulaw,7=alaw,     257=IBM Mu-Law, 258=IBM A-Law, 259=ADPCM
    uint16_t        NumOfChan;      // Number of channels 1=Mono 2=Sterio
    uint32_t        SamplesPerSec;  // Sampling Frequency in Hz
    uint32_t        bytesPerSec;    // bytes per second
    uint16_t        blockAlign;     // 2=16-bit mono, 4=16-bit stereo
    uint16_t        bitsPerSample;  // Number of bits per sample
    /* "data" sub-chunk */
    uint8_t         Subchunk2ID[4]; // "data"  string
    uint32_t        Subchunk2Size;  // Sampled data length
} wav_hdr;
int getFileSize(FILE* inFile);

__global__ void all_in(cufftComplex* in, cufftComplex* out);

int main(int argc, char ** argv) {
    wav_hdr wavHeader;
    int headerSize = sizeof(wav_hdr);

    const char* filePath;
    filePath = argv[1];

    FILE* wavFile = fopen(filePath, "r");
    if (wavFile == nullptr)
    {
        fprintf(stderr, "Unable to open wave file: %s\n", filePath);
        return 1;
    }

    
    //Read the header
    size_t bytesRead = fread(&wavHeader, 1, headerSize, wavFile);
    short data_array[wavHeader.Subchunk2Size/2];

    auto start = std::chrono::system_clock::now();
    if (bytesRead > 0)
    {

        //Read the data
        // uint16_t bytesPerSample = wavHeader.bitsPerSample / 8;      //Number     of bytes per sample
        // uint64_t numSamples = wavHeader.ChunkSize / bytesPerSample; //How many samples are in the wav file?
        int8_t* buffer = new int8_t[BUFFER_SIZE];

        int i = 0;
        while ((bytesRead = fread(buffer, sizeof buffer[0], BUFFER_SIZE / (sizeof buffer[0]), wavFile)) > 0)
        {
            /** DO SOMETHING WITH THE WAVE DATA HERE **/
            memcpy(&data_array[BUFFER_SIZE*i/2], &buffer[0], bytesRead);
            i++;
        }
        delete [] buffer;
        buffer = nullptr;
        printf("%d\n", i);
    }
    fclose(wavFile);





    printf("[simpleCUFFT] is starting...\n");
    // Allocate host memory for the signal
    // cufftComplex* h_signal = (cufftComplex*)malloc(sizeof(cufftComplex) * SIGNAL_SIZE);
    cufftComplex* h_signal = (cufftComplex*) malloc(sizeof(cufftComplex) * SIGNAL_SIZE);


    for(int bigi = 0; bigi < wavHeader.Subchunk2Size/2/SIGNAL_SIZE; bigi++){

        // memcpy(h_signal, &data_array[0], SIGNAL_SIZE);
        for(int i = 0; i < wavHeader.Subchunk2Size/2; i++){
            h_signal[i].x = (float) data_array[bigi*SIGNAL_SIZE + i];
            // h_signal[i].x=100000;
            h_signal[i].y = 0.0f;
        }

        // Initalize the memory for the signal
        int mem_size = sizeof(cufftComplex) * SIGNAL_SIZE;

        // Allocate device memory for signal
        cufftComplex* g_signal;
        cudaMalloc((void**)&g_signal, mem_size);
        // Copy host memory to device




        cudaMemcpy(g_signal, h_signal, mem_size,
                   cudaMemcpyHostToDevice);

        cufftComplex* g_out;
        cudaMalloc((void**)&g_out, sizeof(cufftComplex) * wavHeader.Subchunk2Size/2/SIGNAL_SIZE);

        cufftComplex* h_fft;
        h_fft = (cufftComplex*) malloc(sizeof(cufftComplex) * wavHeader.Subchunk2Size/2/SIGNAL_SIZE);






        // CUFFT plan
        cufftHandle plan;
        cufftPlan1d(&plan, SIGNAL_SIZE, CUFFT_C2C, 1);

        // Transform signal and kernel
        printf("Transforming signal cufftExecC2C\n");
        cufftResult err = cufftExecC2C(plan, (cufftComplex *)g_signal, (cufftComplex *)g_out, CUFFT_FORWARD);    


        // handle the fft
        unit_in<<<1,1>>>()


    }

    // cuda mem copy to host
    cudaMemcpy(h_fft, g_out, sizeof(cufftComplex) * wavHeader.Subchunk2Size/2/SIGNAL_SIZE, 
        cudaMemcpyDeviceToHost);


    // cufftComplex* g_signal_out;
    // cudaMalloc((void**)&g_signal_out, mem_size);

    // // Transform signal back
    // printf("Transforming signal back cufftExecC2C\n");
    // cufftExecC2C(plan, (cufftComplex *)g_out, (cufftComplex *)g_signal_out, CUFFT_INVERSE);


    // // float* h_out = h_signal;
    // cufftComplex* h_out = (cufftComplex*) malloc(sizeof(cufftComplex) * wavHeader.Subchunk2Size/2);
    // cudaMemcpy(h_out, g_signal, mem_size, cudaMemcpyDeviceToHost);


    for(int i = 0; i < SIGNAL_SIZE; i++){
        printf("fft[%d]: %f\n", i, h_fft[i].x);
    }

    // printf("%s\n", "from here hi!");

    // for(int i = 0; i < SIGNAL_SIZE; i++){
    //     printf("fft[%d]: %f\n", i, h_out[i].x);
    // }

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    printf("Time using in CPU is : %f\n", elapsed_seconds);
    // printf("Error info: %s\n", err);

    

    free(h_signal);
    free(h_fft);

    cudaFree(g_signal);
    cudaFree(g_out);
    // cudaFree(g_signal_out);

    return 0;
}

// find the file size
int getFileSize(FILE* inFile)
{
    int fileSize = 0;
    fseek(inFile, 0, SEEK_END);

    fileSize = ftell(inFile);

    fseek(inFile, 0, SEEK_SET);
    return fileSize;
}




__global__ void unit_in(cufftComplex* in, cufftComplex* out){
    int index = threadIdx.x + blockIdx.x * 1024;

    // copy to local memory

    // here: optimized
    cufftComplex local_in[SIGNAL_SIZE];
    cufftComplex local_out[SIGNAL_SIZE];

    for(int i = 0; i < SIGNAL_SIZE; i++){
        local_in[i] = in[i+index*SIGNAL_SIZE];
    }

    cufftHandle plan;
    cufftPlan1d(&plan, SIGNAL_SIZE, CUFFT_C2C, 1);

    // cufftResult err = cufftExecC2C(plan, (cufftComplex *)in[SIGNAL_SIZE*index:SIGNAL_SIZE*(index+1)-1], (cufftComplex *)g_out, CUFFT_FORWARD);    

    
    cufftResult err = cufftExecC2C(plan, (cufftComplex *)&local_in, (cufftComplex *)&local_out, CUFFT_FORWARD);    


    // get biggest FFT
    int k = 0;
    float max_fft_value = (local_out[k].x > 0) ? local_out[k].x:-local_out[k].x;
    for(int i = 0; i < SIGNAL_SIZE / 2; i++){
        float curt = (local_out[i].x > 0) ? local_out[i].x:-local_out[i].x;
        if(curt > max_fft_value){
            k = i;
            max_fft_value = curt;
        }
    }

    float freq = k * 48000/SIGNAL_SIZE;

    out[index].x = freq;
    out[index].y = 0.0f;

    cufftDestroy(plan);

}

