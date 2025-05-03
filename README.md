# Bio-Inspired Optimization Algorithms

This repository contains the code and resources for a **university research-based project** focused on the exploration and comparison of **bio-inspired optimization algorithms**.

## Project Overview

As part of our university coursework, this project investigates the performance, efficiency, and behavior of several popular bio-inspired metaheuristic algorithms. Specifically, we are implementing and analyzing:

- **Marine Predators Algorithm (MPA)**
- **Particle Swarm Optimization (PSO)**
- **Hybrid Levenberg–Marquardt + Improved MPA (LM-IMPA)**

The project aims to compare the strengths and limitations of these algorithms in various optimization scenarios. Our goal is to understand how these methods perform in terms of convergence speed, accuracy, and robustness.

## Algorithms Covered

### Marine Predators Algorithm (MPA)
A metaheuristic inspired by the foraging strategies of marine predators, designed to balance exploration and exploitation through Brownian and Lévy flight mechanisms.

### Particle Swarm Optimization (PSO)
A population-based optimization technique inspired by the collective behavior of bird flocks and fish schools, well-known for its simplicity and adaptability.

### LM-IMPA (Hybrid Algorithm)
A hybrid approach combining the global search power of MPA with the local refinement capabilities of the Levenberg–Marquardt algorithm, designed to improve convergence and accuracy.


### Contribution Statement:

**Ayesha Rahman**  
Main pipeline - Main.py and Plotting.py  
Algorithms - Adam, LM_IMPA, MPA  
Benchmarks - BreastCancer, Sphere, listSort, Rosenbrock, Ackley  

**James Zhangly**  
Algorithms - PSO  
Benchmarks - BreasCancerNN, Rastrigin, Schwefel, Griewank  

> Modifications to shared functions, classes, and methods were made collaboratively to ensure compatibility and maintain a unified structure for the pipeline. All code was written with agreed-upon standards to support seamless integration through a shared `main()` function.

### **Running Instructions**

1. **Execute the main script**:

   ```bash
   python main.py
   ```

2. **Output files**:

   * All generated plots and result files are saved in the `results/plots/` directory.

