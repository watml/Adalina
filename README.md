
This repository contains the code for our paper:  
**Adalina: Adaptive Linear Approximation for the Shapley Value and Beyond**, which can be found at [https://openreview.net/forum?id=hMTCuqjCW8](https://openreview.net/forum?id=hMTCuqjCW8).

---



### Testing and Verification

The `test/` directory contains two scripts used to verify the correctness  of our implementation.


## Replicate our Results

Follow the steps below to  generate the experimental results and reproduce the figures from the paper.


### 1. Run Experiments
Use the `-p` flag followed by the number of available CPUs to parallelize the runs.

- To generate the main experimental results:

  ​```
  python main.py -p <number_of_cpus>
  ​```

- To generate the results used for Figure 2:

  ​```
  python main_vr.py -p <number_of_cpus>
  ​```

### 2. Generate Figures
Once the results are ready, the figures can be plotted using:
```
python plot_adaptive.py
python plot_comparison.py
python plot_ps.py
python plot_vr.py
```

