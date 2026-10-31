# Parameter-Efficient Fine-Tuning for Vietnamese Legal Reasoning: LoRA, LoRA+, DoRA and PiSSA on Qwen3-4B

Course project for **CS431 (Deep Learning Techniques and Applications)**, University of Information Technology, VNU-HCM. We compare parameter-efficient fine-tuning (PEFT) methods on a Vietnamese legal small language model and evaluate legal reasoning with an LLM-as-a-judge protocol. A Vietnamese version of the original course README is in [`docs/README_vi.md`](docs/README_vi.md).

**Team:** Tran Thi Cam Tu (team lead: planned the experiments and ran training and evaluation), Le Ngoc Phuong Thao, Dinh Hoang Phuc.

## Setup

| Item | Value |
|---|---|
| Base model | `VLSP2025-LegalSML/qwen3-4b-legal-pretrain` (Qwen3-4B), 4-bit quantization, Unsloth |
| Training data | `QuangTran276/new_reasoning` (Hugging Face) |
| Test data | `VLSP2025-LegalSML/Public-Test`; results below are on the 144 samples of Task 3 |
| PEFT | rank 32, alpha 64, all linear layers; LoRA, LoRA+ (LR ratio 16), DoRA, PiSSA (SVD init) |
| Optimizer | AdamW 8-bit, 1 epoch, effective batch 128, seed 3407 |
| Learning rate | 2e-5 for LoRA/DoRA; **1e-5 for PiSSA** (see limitations) |

## Results

All numbers are means over 144 generated answers, scored from 1 to 5 by an LLM judge. They come from the files in [`evaluation/results/`](evaluation/results/).

**Judge A: Llama-3.3-70B, Bloom's-taxonomy rubric**

| Method | Recognition | Understanding | Reasoning | Interpretation | Ethics/Bias | **Average** |
|---|---|---|---|---|---|---|
| LoRA | 2.66 | 3.27 | 2.79 | 1.90 | 4.69 | **3.06** |
| LoRA+ | 2.56 | 3.16 | 2.70 | 1.77 | 4.69 | **2.98** |
| PiSSA | 2.66 | 3.27 | 2.78 | 1.85 | 4.67 | **3.05** |

<!-- TODO: confirm which output file each judge file scored (the LoRA+ judge file lists `lora_task3_detailed_responses.json` as its input). -->

**Judge B: Qwen3-32B, legal-answer rubric (PiSSA only)**

| Legal accuracy | Logical structure | Completeness | Relevance | Overall quality | **Average** |
|---|---|---|---|---|---|
| 2.61 | 3.53 | 2.12 | 3.53 | 2.63 | **2.88** |

### What the results say

- The three methods are **statistically indistinguishable** here: average scores differ by at most 0.08 on a 5-point scale, while per-sample standard deviations are 0.6 to 0.9. We do not claim that one method is better.
- All methods are strong on ethics/bias (about 4.7) and weak on legal **interpretation** (about 1.8), so the bottleneck is the model and data, not the PEFT method.
- With Judge B, the PiSSA model is relatively good at logical structure and relevance (3.53) but weak at completeness (2.12) and legal accuracy (2.61).

### Limitations

- **Single run per method** (one seed) and one test split of 144 samples. No confidence intervals.
- **Different learning rates:** PiSSA used 1e-5 while LoRA used 2e-5, so the comparison is not perfectly controlled.
- **LLM judges** can be biased and the two judges use different rubrics, so Tables A and B are not comparable with each other.
- **DoRA:** a configuration is provided (`configs/dora_config.yaml`), but this repository does not contain DoRA evaluation results.
<!-- TODO: add DoRA results if the runs exist, or keep this note. -->

## Repository layout

```
configs/      one YAML per method (lora, loraplus, dora, pissa, qlora, ...) plus shared base config
data/         data preparation script
training/     training entry point and PEFT loading utilities
evaluation/   generation, metrics, LLM-judge scripts and result files
rag/          retrieval-augmented variant (knowledge base, chunker, pipeline)
demo/         demo app
docs/         research proposal, experiment plan, Vietnamese README
```

`docs/vlegal_experiments.md` is a **planned** experiment design with placeholder numbers, not results.

## Reproduce

```bash
pip install -r requirements.txt   # install Unsloth following its official guide for your CUDA/PyTorch
python data/prepare_data.py --input_dir data/raw --output_dir data/processed
python training/train.py --peft_config configs/pissa_config.yaml    # or lora_config.yaml, dora_config.yaml, ...
```

<!-- TODO: add the exact generation and judge commands (see evaluation/run_eval.py and evaluation/llm_judge_task3.py). -->

The judge scripts need API keys (for example a Groq key). Put them in a local `.env` file, which is git-ignored.

## Acknowledgements

Training hyper-parameters follow the Bosch@AI_Team paper noted in `configs/base_config.yaml`; the base model and datasets come from the VLSP 2025 LegalSML organizers. Libraries: Unsloth, Hugging Face Transformers, PEFT, TRL.
