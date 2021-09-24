#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd

res_dt = pd.read_csv('npb/results_nas_dt.txt', delim_whitespace=True, header=None)

res_dt['x'] = res_dt[0] + ', ' + res_dt[2] + ', ' + res_dt[4].astype(str) 

n = res_dt['x']
s = res_dt[5]

plt.figure(figsize=(12, 8))
plt.title('NAS Data Traffic')

line = plt.bar(n,s)
plt.xlabel('Device, Size class, Processes')
plt.ylabel("Runtime, s")

for i in range(len(s)):
    plt.annotate(str(s[i]), xy=(n[i],s[i]), ha='center', va='bottom')

plt.show()
plt.savefig('dt.pdf')

res_dt = pd.read_csv('npb/results_nas_is.txt', delim_whitespace=True, header=None)
res_dt['x'] = res_dt[0] + ', ' + res_dt[2] + ', ' + res_dt[3].astype(str)
n = res_dt['x']
s = res_dt[4]

plt.figure(figsize=(12, 8))
plt.title('NAS Integer Sort')


line = plt.bar(n,s)
plt.xlabel('Device, Size class, Processes')
plt.ylabel("Runtime, s")

for i in range(len(s)):
    plt.annotate(str(s[i]), xy=(n[i],s[i]), ha='center', va='bottom')

plt.show()
plt.savefig('is.pdf')

res_dt_cpu = pd.read_csv('pi/plot_pi_cpu.txt', delim_whitespace=True, header=None)
res_dt_gpu = pd.read_csv('pi/plot_pi_gpu.txt', delim_whitespace=True, header=None)
res_dt = pd.concat([res_dt_cpu, res_dt_gpu])
res_dt['x'] = res_dt[0] + ', ' + res_dt[1].astype(str)
n = res_dt['x']
s = res_dt[2]

plt.figure(figsize=(12, 8))
plt.title('Pi')

line = plt.bar(n,s)
plt.xlabel('Device, Processes')
plt.ylabel("Runtime, s")

for i in range(len(s)):
    plt.annotate(str(s[i]), xy=(n[i],s[i]), ha='center', va='bottom')

plt.show()
plt.savefig('pi.pdf')

res_dt = pd.read_csv('gpu_mpi-sputnipic/sputnipic_result.txt', delim_whitespace=True, header=None)
res_dt['x'] = res_dt[0] + ', ' + res_dt[1].astype(str)
n = res_dt['x']
s = res_dt[2]

plt.figure(figsize=(12, 8))
plt.title('SputniPIC')

line = plt.bar(n,s)
plt.xlabel('Benchmark, Processes')
plt.ylabel("Runtime, s")

for i in range(len(s)):
    plt.annotate(str(s[i]), xy=(n[i],s[i]), ha='center', va='bottom')

plt.show()
plt.savefig('sputnipic.pdf')

