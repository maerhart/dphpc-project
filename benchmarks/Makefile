
.PHONY: all
all: 

benchmark_pi_cpu.txt: benchmark_pi_cpu.sh
	bash benchmark_pi_cpu.sh

benchmark_pi_gpu.txt: benchmark_pi_gpu.sh
	bash benchmark_pi_gpu.sh

benchmark_is_cpu.txt: benchmark_is_cpu.sh
	bash benchmark_is_cpu.sh

benchmark_is_gpu.txt: benchmark_is_gpu.sh
	bash benchmark_is_gpu.sh

benchmark_dt_cpu.txt: benchmark_dt_cpu.sh
	bash benchmark_dt_cpu.sh

benchmark_dt_gpu.txt: benchmark_dt_gpu.sh
	bash benchmark_dt_gpu.sh

benchmark_sputnipic_cpu.txt: benchmark_sputnipic_cpu.sh
	bash benchmark_sputnipic_cpu.sh

benchmark_sputnipic_gpu.txt: benchmark_sputnipic_gpu.sh
	bash benchmark_sputnipic_gpu.sh

.PHONY: benchmark_cpu
benchmark_cpu: benchmark_pi_cpu.txt benchmark_is_cpu.txt benchmark_dt_cpu.txt benchmark_sputnipic_cpu.txt

.PHONY: benchmark_gpu
benchmark_gpu: benchmark_pi_gpu.txt benchmark_is_gpu.txt benchmark_dt_gpu.txt benchmark_sputnipic_gpu.txt

.PHONY: clean
clean:
	rm -f benchmark*txt