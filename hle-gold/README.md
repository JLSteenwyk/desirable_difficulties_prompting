---
dataset_info:
  features:
  - name: id
    dtype: string
  - name: question
    dtype: string
  - name: image
    dtype: string
  - name: image_preview
    dtype: image
  - name: answer
    dtype: string
  - name: answer_type
    dtype: string
  - name: author_name
    dtype: string
  - name: rationale
    dtype: string
  - name: rationale_image
    dtype: image
  - name: raw_subject
    dtype: string
  - name: category
    dtype: string
  - name: canary
    dtype: string
  splits:
  - name: train
    num_bytes: 16938456.0072
    num_examples: 149
  download_size: 4982881
  dataset_size: 16938456.0072
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
license: mit
language:
- en
task_categories:
- question-answering
tags:
- biology
- chemistry
---


# Humanity's Last Exam (HLE) Bio/Chem Gold

[Humanity’s Last Exam (HLE)](https://lastexam.ai/paper) is a challenging question-answering AI benchmark covering advanced academic fields including Math, Physics, Chemistry, Biology, Engineering, and Computer Science.

At [FutureHouse](https://www.futurehouse.org/), we audited the biology and chemistry subsets of HLE using a combination of expert human evaluators and our in-house research agent, and found that around 30% of the questions contain answers directly contradicted by peer-reviewed literature.

To address this, we created HLE Bio/Chem Gold, a validated subset of HLE biology and chemistry scientifically sound questions, to support more reliable benchmarking in these disciplines.

For more information about our analysis, see our 📄 [blog](https://www.futurehouse.org/research-announcements/hle-exam).

#### Load the dataset:

```python
from datasets import load_dataset

dataset = load_dataset("futurehouse/hle-gold-bio-chem", split="train")
```