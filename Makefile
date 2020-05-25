test: test.cu
	nvcc -o $@ $< -lcublas -lnvidia-ml

clean:
	rm -f test
