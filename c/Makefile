git:
	git add .
	git commit -m "auto"
	git push


cuda:
	nvcc -arch=sm_35 -o main.out main.cu -rdc=true -lcudadevrt -std=c++11 -lcufft


cuda_ss:
	nvcc -arch=sm_35 -o main.out main_pseudo.cu -rdc=true -lcudadevrt -std=c++11 -lcufft	

fft:
	nvcc -arch=sm_35 -o main.out fft.cu -rdc=true -lcudadevrt -std=c++11 -lcufft

fft2:
	nvcc -arch=sm_35 -o main.out fft2.cu -rdc=true -lcudadevrt -std=c++11 -lcufft


main:
	g++ -std=c++11 -lrfftw -lfftw -lm main.cpp -o main.out
	./main.out part_output.wav


reader:
	g++ -std=c++11 WaveReader.cpp -o WaveReader.out
	./WaveReader.out part_output.wav