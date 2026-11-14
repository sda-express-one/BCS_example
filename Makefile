CXX      = g++
CXXFLAGS = -O3 -march=native -fopenmp -std=c++17 -Wall -Wextra
CONDA    ?= $(shell command -v conda 2>/dev/null || echo $(HOME)/miniconda3/bin/conda)

bcs_swave: bcs_swave.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

run: bcs_swave
	./bcs_swave

plot:
	$(CONDA) run -n env00 python plot_swave.py

clean:
	rm -f bcs_swave

.PHONY: run plot clean
