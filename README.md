# Is Your Video Language Model a Reliable Judge?

This repository contains the code for our paper ["Is Your Video Language Model a Reliable Judge?"](https://openreview.net/forum?id=m8yby1JfbU).

## Abstract

This work investigates the reliability of using Video Language Models (VLMs) as judges to evaluate other VLMs. We examine collective judgment approaches with mixed reliability models and explore factors impacting evaluation reliability. Our findings challenge existing automatic VLM evaluation methods and highlight the need for more advanced evaluation techniques that account for individual model reliability.

## Key Findings

- **Collective judgments from mixed reliability models do not necessarily improve accuracy** - Inclusion of less reliable judges can introduce noise into the evaluation process
- **Improved model understanding alone is insufficient for reliable evaluation** - We demonstrate this by fine-tuning an underperforming VLM (Video-LLaVA)
- **Individual model reliability matters** - Evaluation methods must account for the varying reliability of different VLM judges

## Methodology

### Evaluation Framework

Our evaluation framework builds on LMMS-Eval and implements several key innovations:

1. **Multi-Judge Evaluation**: We evaluate 5 candidate VLMs (LLaMA-Vid, GPT-4V, Video-ChatGPT, mPlug-Owl-Video, Video-LLaVA) using 4 judge models (Video-LLaVA, LLaMA-Vid, GPT-4V, InternVL2)

2. **Debate-Based Consensus**: Implemented a debate mechanism between GPT-3.5 and GPT-4o judges to reach consensus on evaluations (see `src/logs_phase1/correct/correct_debate.py`)

3. **Comprehensive Benchmarks**: Evaluation across multiple dimensions using CVRR (Comprehensive Video Reasoning and Robustness) dataset with 11 distinct categories:
   - Continuity and Object Instance Count
   - Fine-grained Action Understanding
   - Interpretation of Social Context
   - Interpretation of Visual Context
   - Multiple Actions in a Single Video
   - Non-existent Actions (with/without scene depictions)
   - Partial Actions
   - Time Order Understanding
   - Understanding of Emotional Context
   - Unusual and Physically Anomalous Activities

### Datasets Used

- **CVRR-ES**: Comprehensive Video Reasoning and Robustness Evaluation Suite
- **VideoChatGPT**: Benchmark for video understanding


## Repository Structure

```
vlm_judge/
├── src/
│   ├── lmms_eval/           # Core evaluation framework
│   │   ├── api/              # API interfaces for model evaluation
│   │   ├── models/           # VLM model implementations
│   │   ├── tasks/            # Evaluation task definitions
│   │   ├── filters/          # Data filtering and transformation
│   │   └── evaluator.py     # Main evaluation logic
│   │
│   ├── logs_phase1/          # Phase 1 experimental logs
│   │   ├── correct/          # Correction and debate experiments
│   │   └── create_data.py   # Data preparation scripts
│   │
│   ├── logs_phase2/          # Phase 2 experimental logs
│   ├── logs_phase2_processed_finetune/  # Fine-tuning processed data
│   ├── logs_phase3/          # Phase 3 experimental logs
│   ├── logs_phase3.1/        # Extended Phase 3 experiments
│   │
│   ├── tools/                # Utility tools
│   │   ├── live_bench/       # Live benchmark implementation
│   │   └── make_vatex.py    # VATEX dataset preparation
│   │
│   ├── miscs/                # Miscellaneous utilities
│   └── docs/                 # Documentation
```

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 2.1.0
- CUDA-compatible GPU recommended

### Setup

1. Clone the repository:
```bash
git clone https://github.com/pkulium/vlm_judge.git
cd vlm_judge
```

2. Install dependencies:
```bash
cd src
pip install -e .
```

The package will install all required dependencies including:
- Transformers, PyTorch, torchvision
- Evaluation frameworks (datasets, evaluate)
- Video processing libraries (decord, av, opencv-python-headless)
- And other dependencies specified in `pyproject.toml`

## Usage

### Running Evaluations

The main evaluation script can be run using:

```bash
lmms-eval --model <model_name> --tasks <task_name> --output_path <output_dir>
```

### Experiment Phases

Our experiments are organized in multiple phases:

1. **Phase 1**: Initial reliability assessment of VLM judges
   - Run correction experiments: `python src/logs_phase1/correct/correct.py`
   - Run debate experiments: `python src/logs_phase1/correct/correct_debate.py`

2. **Phase 2**: Fine-tuning experiments with Video-LLaVA
   - Process fine-tuning data: `python src/logs_phase2_processed_finetune/create_data.py`

3. **Phase 3 & 3.1**: Extended reliability analysis with collective judgments
   - Process evaluation data: `python src/logs_phase3/create_data.py`

### Data Preparation

To prepare datasets for llm judge evaluation:

```python
python src/logs_phase*/create_data.py
```


## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{liu2025vlmjudge,
  title={Is Your Video Language Model a Reliable Judge?},
  author={Liu, Ming and Zhang, Wensheng},
  booktitle={ICLR 2025},
  year={2025},
  url={https://openreview.net/forum?id=m8yby1JfbU}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](src/LICENSE) file for details.


## Acknowledgments

This work builds upon the LMMS-Eval framework for evaluating large multi-modality language models. We thank the community for their contributions to advancing VLM evaluation methodologies.
