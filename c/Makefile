git:
	git add .
	git commit -m "auto"
	git push


cuda:
	nvcc -arch=sm_35 -o main.out main.cu -rdc=true -lcudadevrt -std=c++11 -lcufft


fft:
	nvcc -arch=sm_35 -o main.out fft1.cu -rdc=true -lcudadevrt -std=c++11 -lcufft


main:
	g++ -std=c++11 main.cpp -o main.out
	./main.out test.wav


reader:
	g++ -std=c++11 WaveReader.cpp -o WaveReader.out
	./WaveReader.out part_output.wav