#include <stdio.h>
#include <math.h>
#include <ctime>
#include <chrono>
#include <cufft.h>

#define BUFFER_SIZE 4096

// Complex data type
typedef float2 Complex;

#define SIGNAL_SIZE 1000

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
    float data_array[wavHeader.Subchunk2Size];
    if (bytesRead > 0)
    {

        //Read the data
        // uint16_t bytesPerSample = wavHeader.bitsPerSample / 8;      //Number     of bytes per sample
        // uint64_t numSamples = wavHeader.ChunkSize / bytesPerSample; //How many samples are in the wav file?
        float* buffer = new float[BUFFER_SIZE];

        int i = 0;
        while ((bytesRead = fread(buffer, sizeof buffer[0], BUFFER_SIZE / (sizeof buffer[0]), wavFile)) > 0)
        {
            /** DO SOMETHING WITH THE WAVE DATA HERE **/
            memcpy(&data_array[BUFFER_SIZE*i], &buffer[0], bytesRead);
            i++;
        }
        delete [] buffer;
        buffer = nullptr;
        printf("%d\n", i);

    }
    fclose(wavFile);

    // int8_t *ginit_array;
    // cudaMalloc((void **) &ginit_array, wavHeader.Subchunk2Size);
    // cudaMemcpy(ginit_array, data_array, wavHeader.Subchunk2Size, cudaMemcpyHostToDevice);

    // int8_t *gout_array;
    // cudaMalloc((void **) &gout_array, wavHeader.Subchunk2Size/ BUFFER_SIZE);
    // cuda_fft<<<1, ceil(wavHeader.Subchunk2Size/BUFFER_SIZE)>>>(ginit_array, gout_array);
    // printf("%s\n", "done");

    printf("[simpleCUFFT] is starting...\n");
    // Allocate host memory for the signal
    // Complex* h_signal = (Complex*)malloc(sizeof(Complex) * SIGNAL_SIZE);
    // Initalize the memory for the signal

    // R 2 C ?
    // 




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
