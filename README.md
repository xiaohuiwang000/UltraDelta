# Breaking the Compression Ceiling: Data-Free Pipeline for Ultra-Efficient Delta Compression

This repository is the official implementation of **"Breaking the Compression Ceiling: Data-Free Pipeline for Ultra-Efficient Delta Compression"**.

📢 *This paper has been accepted at NeurIPS 2025.*

<p align="center">
  <img src="asset\ultradelta.png" alt="UltraDelta Pipeline Overview" width="700">
</p>

---

## 📦 Install

To set up the environment and install dependencies:

```bash
conda create -n ultradelta_vit python=3.10
conda activate ultradelta_vit

git clone https://github.com/xiaohuiwang000/UltraDelta.git
cd UltraDelta

pip install -r requirements.txt
```

## 📂 Datasets and Checkpoints

Please follow the AdaMerging repository for detailed instructions on downloading datasets:
🔗 [AdaMerging # Datasets](https://github.com/EnnengYang/AdaMerging?tab=readme-ov-file#datasets)


You can download the fine-tuned checkpoints from the **Task Vectors** repository:  
🔗 [task_vectors # checkpoints](https://github.com/mlfoundations/task_vectors#checkpoints)

The corresponding Google Drive folder is available here:  
🔗 [task_vectors_checkpoints](https://drive.google.com/drive/folders/1u_Tva6x0p6oxu5Eo0ZZsf-520Cc_3MKw)

```text
UltraDelta/
├── data/  
│   ├── dtd/
│   │   ├── test/
│   │   │   ├── banded/
│   │   │   ├── blotchy/
│   │   │   ├── ...
│   │   ├── ...
│   ├── EuroSAT_splits/
│   ├── gtsrb/
|   ├── ...
│
├── checkpoints/ 
│   ├── ViT-B-32/
│   │   ├── Cars/
│   │   │   ├── finetuned.pt
│   │   ├── DTD/
│   │   │   ├── finetuned.pt
│   │   ├── ...
│   ├── ViT-L-14/
│   │   ├── ...
│
└── ...
```

## 🚀 Evaluation

To evaluate, simply run:

```bash
bash run.sh
```

## 🧩 TODO

- [ ] Integrate **LLM compression pipeline** (e.g., LLaMA, Qwen)
- [ ] Integrate **NLP models compression pipeline** (e.g., T5, RoBERTa)


## 🧠 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{wang2025ultradelta,
  title     = {Breaking the Compression Ceiling: Data-Free Pipeline for Ultra-Efficient Delta Compression},
  author    = {Wang, Xiaohui and Ye, Peng and Huang, Chenyu and Zheng, Shenghe and Zhang, Bo and Bai, Lei and Ouyang, Wanli and Chen, Tao},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025},
  url       = {https://proceedings.neurips.cc/paper_files/paper/2025/file/21912f7057935149fa58408ee8cb460e-Paper-Conference.pdf}
}
```

## 🙏 Acknowledgement

Our implementation references the following excellent open-source projects — many thanks to their authors:

- **Task Arithmetic**: [https://github.com/mlfoundations/task_vectors](https://github.com/mlfoundations/task_vectors)  
- **AdaMerging**: [https://github.com/EnnengYang/AdaMerging](https://github.com/EnnengYang/AdaMerging)
- **EMR-Merging**: [https://github.com/harveyhuang18/EMR_Merging
](https://github.com/harveyhuang18/EMR_Merging)
