<h1 align="center">Geometric Algebra for Rigid Body Dynamics</h1>
<p align="center">Analytical and Computational Solution of a Symmetric Top Using Clifford Algebra</p>

This repository contains the implementation of the experiments and mathematical formulation presented in the article _"Geometric Algebra as a Language for Physics: Application to Rigid Body Dynamics"_ by David Muñoz Hernández.

The work explores the power of **geometric algebra** as a computational and conceptual tool for simulating **rotational motion**, particularly focusing on the symmetric top using **rotor-based transformations** and the **Clifford Python library**.

## Contents

This repository is organized as follows:

- `symmetric_top_solver.py`: Main Python script with detailed step-by-step explanation and comments.
- `figures/`: Contains simulation results and visualizations.
- `media/`: Folder for the animation showing the top's precessional motion.
- `TFG.pdf`: Full article describing the mathematical foundations and physical insights.

The solution involves two simulations with different initial angular momentum vectors to demonstrate:
1. Motion around the principal axis of inertia.
2. Precession and energy conservation under slight perturbation.

## Requirements

To run this project, install the following packages:

- Python ≥ 3.8  
- [Clifford](https://github.com/pygae/clifford) ≥ 1.4  
- NumPy ≥ 1.21  
- Matplotlib ≥ 3.4  
- SciPy ≥ 1.7  
- tqdm (for progress bars)
- imageio (for the animation)

## Installation and use

You can install everything using:

```
pip install clifford numpy scipy matplotlib tqdm imageio
```

### How to Run the Simulation
Clone the repository:
```
git clone https://github.com/cloudlab-aia/geometric-algebra-top.git
cd geometric-algebra-top
```

Run the main script:
```
python symmetric_top_solver.py
```

Output:

Plots of angular velocity and kinetic energy will be saved in figures/.

An animation will be generated showing the motion of the symmetric top.

## Simulation Results
### First Simulation
- Initial angular momentum aligned with the z-axis.

- The motion shows constant rotation with no precession.

- Kinetic energy remains constant (verified numerically).

###Second Simulation
- Slight yz-component added to angular momentum.

- Precession observed in the I₃ axis and energy remains conserved.

## Animation
Below is a preview of the animation generated during simulation:

<p align="center"> <img src="media/symmetric_top_animation.gif" width="480"/> </p>
## Citation
```bibtex
@article{munoz_david_PINN_2025,
	title = {Geometric Algebra as a Language for Physics: Application to Rigid Body Dynamics,
	issn = {},
	journal = {},
	author = {e},
	year = {2025},
	pages = {},
	note = {in press},
}
```

## Acknowledgements
This research has been performed for the research project <a href="https://aia.ua.es/en/proyectos/federated-serverless-architectures-for-heterogeneous-high-performance-computing-in-smart-manufacturing.html" target="_blank">Federated Serverless Architectures for Heterogeneous High Performance Computing in Smart Manufacturing</a>, at the Applied Intelligent Architectures Research Group of the University of Alicante (Spain).

## License
This project is licensed under the <a href="LICENSE.txt">GPL-3 license</a>.
