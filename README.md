# CelerisAi

<p align="center">
  <img src="https://github.com/wrenteria/CelerisAi/blob/main/docs/source/images/CelerisAILogo.png" alt="CelerisAI Logo" width="400">
</p>

<p align="center">
    <img src="https://img.shields.io/github/license/wrenteria/CelerisAi" alt="Github License">
    <img src="https://img.shields.io/github/stars/wrenteria/CelerisAi" alt="Github stars">
    <img src="https://img.shields.io/github/forks/wrenteria/CelerisAi" alt="Github forks">
</p>

CelerisAi is a Python-[Taichi](https://github.com/taichi-dev/taichi)-based software designed for nearshore wave modeling. This solver offers high-performance simulations on various hardware platforms and seamlessly integrates with machine learning and artificial intelligence environments. The solver leverages the flexibility of Python for customization and interoperability, while Taichi's high-performance parallel programming capabilities ensure efficient computations.

## Key Features
* High Performance: CelerisAi delivers efficient simulations on a wide range of hardware, from personal laptops to high-performance computing clusters.
* Machine Learning Integration: The solver's seamless integration with AI frameworks empowers users to develop hybrid models and leverage data-driven approaches.
* Flexibility and Customization: CelerisAi's Python-based implementation provides a high degree of customization and adaptability to meet specific modeling needs.
* Interoperability: The solver's compatibility with various Python libraries and tools enhances its versatility and integration capabilities.

## Applications
CelerisAi can be applied to a variety of coastal engineering problems, including:
- Wave propagation modeling
- Coastal hydrodynamics
- Tsunami simulation
- Inverse problems (e.g., depth inversion)
- Assimilation problems

## Quick Start
Use the editable install so the `CelerisAI` package is available from any working directory.

```bash
git clone https://github.com/wrenteria/CelerisAi.git
cd CelerisAi
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Documentation
The full user guide, API references, and examples live in the Sphinx docs under `docs/`. Once GitHub Pages is enabled, add the hosted documentation URL here.

## Examples
After installing, run one of the sample simulations:

```bash
python setrun_1D.py
```

### Using CelerisWEBGPU configuration
[Balboa beach](./setrun_web.py)

![Balboa](docs/source/images/Balboa.gif)
```bash
python setrun_web.py
```

## Dependencies
### Core requirements
Installed automatically by `pip install -e .`:

- Python 3.x
- Taichi
- NumPy
- SciPy
- Matplotlib
- ImageIO

### Optional requirements
PyTorch is used for AI integration workflows. Install it separately if you plan to run the differentiability or learning examples.


## License
CelerisAi is released under the MIT License. See [LICENSE](https://github.com/wrenteria/CelerisAi/blob/main/LICENSE) for details.

## Contributions
CelerisAi is an open-source project. Contributions from the community are welcome. Please refer to the project's guidelines for contributing.

## Citation
If you find this version of CelerisAI useful for your research, consider citing:

```
@article{Renteria_2025,
title={CelerisAi: A Nearshore Wave Modeling Framework for Integrated AI Applications},
url={http://dx.doi.org/10.22541/essoar.174129311.11936719/v1},
DOI={10.22541/essoar.174129311.11936719/v1},
publisher={Authorea, Inc.},
author={Renteria, Willington and Lynett, Patrick and Bonus, Justin and Mccann, Maile and Ebrahimi, Behzad},
year={2025},
month=mar }
```
