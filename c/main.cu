#include <stdio.h>
#include <math.h>
#include <ctime>
#include <chrono>
#include <cufft.h>

#define BUFFER_SIZE 4096

// cufftComplex data type
// typedef float2 cufftComplex;

#define SIGNAL_SIZE 4096

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
    cufftComplex* h_signal = (cufftComplex*) malloc(sizeof(cufftComplex) * wavHeader.Subchunk2Size/2);

    // memcpy(h_signal, &data_array[0], SIGNAL_SIZE);
    for(int i = 0; i < wavHeader.Subchunk2Size/2; i++){
        h_signal[i].x = (float) data_array[i];
        // h_signal[i].x = 1.0f;
        h_signal[i].y = 0.0f;
    }

    // for(int i = 0; i < SIGNAL_SIZE; i++){
    //     printf("%f\n", h_signal[i].x);
    // }

    // Initalize the memory for the signal
    int mem_size = sizeof(cufftComplex) * wavHeader.Subchunk2Size/2;

    // Allocate device memory for signal
    cufftComplex* g_signal;
    cudaMalloc((void**)&g_signal, mem_size);
    // Copy host memory to device
    cudaMemcpy(g_signal, h_signal, mem_size,
               cudaMemcpyHostToDevice);

    cufftComplex* g_fft_out;
    cudaMalloc((void**)&g_fft_out, sizeof(cufftComplex) * wavHeader.Subchunk2Size/2);



    // CUFFT plan
    cufftHandle plan;
    int n[1] = {SIGNAL_SIZE};
    cufftResult res = cufftPlanMany(&plan, 1, n,
        NULL, 1, SIGNAL_SIZE,  //advanced data layout, NULL shuts it off
        NULL, 1, SIGNAL_SIZE,  //advanced data layout, NULL shuts it off
        CUFFT_C2C, wavHeader.Subchunk2Size/2/SIGNAL_SIZE);    
        // CUFFT_C2C, 3);


    // Transform signal and kernel
    // printf("Transforming signal cufftExecC2C\n");
    cufftResult err = cufftExecC2C(plan, (cufftComplex *)g_signal, (cufftComplex *)g_fft_out, CUFFT_FORWARD);    


    
    cufftComplex* g_fft_max_out;
    cudaMalloc((void**)&g_fft_max_out, sizeof(cufftComplex) * (wavHeader.Subchunk2Size/2/SIGNAL_SIZE + 1));

    int blockSize = wavHeader.Subchunk2Size/SIGNAL_SIZE/2048 + 1;
    all_in<<<blockSize, SIGNAL_SIZE>>>(g_fft_out, g_fft_max_out);
    

    // cuda mem copy to host
    cufftComplex* h_fft;
    h_fft = (cufftComplex*) malloc(sizeof(cufftComplex) * (wavHeader.Subchunk2Size/2/SIGNAL_SIZE + 1));

    cudaMemcpy(h_fft, g_fft_out, sizeof(cufftComplex) * (wavHeader.Subchunk2Size/2/SIGNAL_SIZE + 1), 
        cudaMemcpyDeviceToHost);


    // cufftComplex* g_signal_out;
    // cudaMalloc((void**)&g_signal_out, mem_size);

    // // Transform signal back
    // printf("Transforming signal back cufftExecC2C\n");
    // cufftExecC2C(plan, (cufftComplex *)g_fft_out, (cufftComplex *)g_signal_out, CUFFT_INVERSE);


    // // float* h_out = h_signal;
    // cufftComplex* h_out = (cufftComplex*) malloc(sizeof(cufftComplex) * wavHeader.Subchunk2Size/2);
    // cudaMemcpy(h_out, g_signal, mem_size, cudaMemcpyDeviceToHost);


    // for(int i = (wavHeader.Subchunk2Size/2 - SIGNAL_SIZE); i < (wavHeader.Subchunk2Size/2); i++){
    //     printf("fft[%d]: %f\n", i, h_fft[i].x);
    // }


    for(int i = 0; i < 24; i++){
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


    cufftDestroy(plan);
    cudaFree(g_signal);
    cudaFree(g_fft_out);
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




__global__ void all_in(cufftComplex* in, cufftComplex* out){
    int index = threadIdx.x + blockIdx.x * 1024;

    // copy to local memory

    // here: optimized
    cufftComplex local_in[SIGNAL_SIZE];
    // cufftComplex local_in[SIGNAL_SIZE];

    for(int i = 0; i < SIGNAL_SIZE; i++){
        local_in[i] = in[i+index*SIGNAL_SIZE];
    }

    // get biggest FFT
    int k = 0;
    float max_fft_value = (local_in[k].x > 0) ? local_in[k].x:-local_in[k].x;
    for(int i = 0; i < SIGNAL_SIZE / 2; i++){
        float curt = (local_in[i].x > 0) ? local_in[i].x:-local_in[i].x;
        if(curt > max_fft_value){
            k = i;
            max_fft_value = curt;
        }
    }

    float freq = k * 48000/SIGNAL_SIZE;

    out[index].x = freq;
    out[index].y = 0.0f;

}

