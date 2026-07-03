# [Edge-Efficient Transformer for End-to-End RF Spectrum Monitoring](https://ieeexplore.ieee.org/document/11593812/authors#authors)

<p align="center">
  <img src="figures/acc_vs_snr.png" alt="Accuracy vs SNR" width="100%">
</p>

Code repository for **E-SpecFormer** (Edge Spectrum monitoring Transformer), a super-lightweight Transformer family for end-to-end automatic modulation and covert channel recognition. 

E-SpecFormer utilizes Conv-Tokenizer and introduces **LiTAN** (Linear Tanh Attention Network), a Softmax- and LayerNorm-free attention mechanism that reduces computational complexity to linear time while increasing accuracy for RF tasks. The architecture is natively hardware-friendly and is designed for HW/SW co-design, enabling high-speed streaming inference of real-time spectrum intelligence and heterogeneous computing (CPU/FPGA) for extreme edge computing.

---

## Models & Performance

The models have been extensively tested on the **RadioML2018** (AMR) and **HT-CC** (Hardware-Trojan-based Covert Channel) datasets. We provide four scalable variants: Nano (N), Small (S), Medium (M), and Large (L).

### RadioML2018 (AMR)
**Input Shape:** `(B, 1, 2, 1024)`

| Model | Avg. Accuracy (SNR > 0 dB) | ARM CPU* (ms) | FPGA** (µs) | Params.| kMACs | Pre-trained Weights |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| E-SpecFormer-N | 86.5% | 0.33 | 124 | 8,314 | 501 | [nano.pt](models/amr/Nano.pt) |
| E-SpecFormer-S | 89.4% | 0.52 | 243 | 14,780 | 924 | [small.pt](models/amr/Small.pt) |
| E-SpecFormer-M | 92.7% | 1.55 | 481 | 36,032 | 4,589 | [medium.pt](models/amr/Medium.pt) |
| E-SpecFormer-L | 94.0% | 3.06 | 719 | 116,532 | 15,040 | [large.pt](models/amr/Large.pt) |

### HT-CC
**Input Shape:** `(B, 1, 2, 640)`

| Model | Avg. Accuracy (SNR > 0 dB) | ARM CPU* (ms) | FPGA** (µs) | Params.| kMACs | Pre-trained Weights |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| E-SpecFormer-N | 94.2% | 0.24 | 92 | 7,687 | 312 | [nano.pt](models/htcc/Nano.pt) |
| E-SpecFormer-S | 95.7% | 0.38 | 181 | 14,153 | 577 | [small.pt](models/htcc/Small.pt) |
| E-SpecFormer-M | 96.1% | 1.04 | 359 | 35,405 | 2,854 | [medium.pt](models/htcc/Medium.pt) |
| E-SpecFormer-L | 96.9% | 2.04 | 537 | 115,601 | 9,355 | [large.pt](models/htcc/Large.pt) |

> \* **ARM CPU Latency:** Evaluated with ONNX Runtime on ARM Cortex-A76 (Raspberry Pi 5).  
> \*\* **FPGA Latency:** Evaluated on Xilinx Zynq MPSoC ZCU104 FPGA (ARM Cortex-A53 CPU + Custom Accelerator).

---

## Usage 

E-SpecFormer processes raw I/Q samples end-to-end. Ensure your PyTorch dataloader yields inputs in the shape `(B, 1, 2, L)`, where `B` is the batch size and `L` is the sequence length (`1024` for RadioML2018, `640` for HT-CC).

### Initialization via Configuration
The cleanest way to build a model variant is using the provided YAML configuration files. This automatically scales the sequence lengths, attention heads, and expansion ratios.

```python
import torch
from especformer.builder import load_config, build_model_from_config

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load configuration for the Nano variant (make sure to change the num_classes)
config = load_config('configs/nano.yaml')

# Automatically build the model
model = build_model_from_config(config, device=device)
```

### Manual Initialization

You an manually instantiate the model by passing the hyperparameters:

```python
from especformer.model import ESpecFormer

# Initialize the Nano variant for RadioML2018 (24 classes) manually
model = ESpecFormer(
    num_classes=24, 
    d_model=32, 
    nhead=8, 
    num_layers=1, 
    dim_feedforward=32, 
    k=16, 
    s=16, 
    dtanh=True, 
    linattn=True
).to(device)
```

### Loading Pre-trained Weights 

Once the model is built, you can load the pre-trained weights from the `models/` directory to test via PyTorch.

```python
# Load the pre-trained state dictionary
model.load_state_dict(torch.load('models/amr/Nano.pt', map_location=device))
model.eval()
```

You can also export the model to ONNX Runtime or use and modify the C source code to compile and run on any CPU for inference test and mathematical evaluation only.

---


## Citation

If you use E-SpecFormer or find this work useful in your research, please consider citing our paper:

> Z. Song, H.-G. Stratigopoulos and H. Aboushady, "Edge-Efficient Transformer for End-to-End RF Spectrum Monitoring," in *IEEE Transactions on Circuits and Systems II: Express Briefs*, 2026, doi: 10.1109/TCSII.2026.3709389.


---

Copyright &copy; 2026 Sorbonne Université, Centre National de la Recherche Scientifique (CNRS), Laboratoire d'Informatique de Paris 6 (LIP6).

Licensed under the [Apache 2.0 License](LICENSE).
