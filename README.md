# Generative AI for PPE (Personal Protective Equipment) Detection

This repository contains a collection of Generative Artificial Intelligence models used for detecting Personal Protective Equipment (PPE) in images. The project's goal is to compare different Vision Language Models (VLM) and object detection models to evaluate their performance in the task of PPE detection, both with and without fine-tuning.

## 📖 Table of Contents

- [Introduction](#introduction)
- [Models Used](#models-used)
  - [YOLOv8](#yolov8)
  - [Phi-3.5 Vision Instruction](#phi-35-vision-instruction)
  - [Molmo-7B-0-0924 VisionLM](#molmo-7b-0-0924-visionlm)
  - [LLaVA-1.6-Mistral](#llava-16-mistral)
  - [GPT-4o and GPT-4o-mini](#gpt-4o-and-gpt-4o-mini)
- [Prerequisites](#prerequisites)
- [Results](#results)
- [Contact](#contact)

## Introduction

Automatic detection of PPE is crucial in various industrial sectors to ensure workplace safety. This project explores the use of Generative Artificial Intelligence models for recognizing PPE in images, comparing the performance of different pre-trained models, some of which do not require fine-tuning to achieve excellent results.

## Models Used

### YOLOv8

- **File**: `yolov8.ipynb`
- **Description**: YOLOv8 is an object detection model that has been fine-tuned on a specific PPE dataset.
- **Performance**: Achieved an accuracy higher than 80% in the PPE detection task.
- **Hardware Used**: NVIDIA GeForce RTX 3070 Ti GPU.

### Phi-3.5 Vision Instruction

- **File**: `phi-3.5-vision.ipynb`
- **Description**: Vision Language Model (VLM) from Google with 4.2 billion parameters.
- **Performance**: Achieves excellent accuracy without any fine-tuning.
- **Hardware Used**: NVIDIA L4 GPU with 24 GB of memory.

### Molmo-7B-0-0924 VisionLM

- **File**: `molmo-7b-visionlm.ipynb`
- **Description**: Optimized VLM model with 7 billion parameters, executable on GPUs with limited memory thanks to the use of `bfloat16`.
- **Performance**: Obtains outstanding results without any fine-tuning in the PPE detection task.
- **Hardware Used**: NVIDIA Tesla P100 GPU with 16 GB of memory.

### LLaVA-1.6-Mistral

- **File**: `llava-1.6-mistral.ipynb`
- **Description**: VLM model that provides excellent results without fine-tuning on the same dataset used for the other models.
- **Performance**: Demonstrates accuracy and computational efficiency.
- **Hardware Used**: NVIDIA Tesla P100 GPU with 16 GB of memory.

### GPT-4o and GPT-4o-mini

- **Files**: `gpt-4o.ipynb` and `gpt-4o-mini.ipynb`
- **Description**: Highly performant VLM models, surpassing even YOLO in their field.
- **Performance**: Achieve excellent results in PPE recognition without the need for specific hardware.
- **Implementation**: Use HTTP requests through APIs provided by OpenAI, eliminating the need for local computational resources.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)
- CUDA-compatible GPU (for local execution of models)
- OpenAI account with API access (for GPT-4o and GPT-4o-mini)

## Results

The models have shown the following performance in the PPE detection task:

- **YOLOv8**: Accuracy higher than 80% after fine-tuning on a specific dataset.
- **Phi-3.5 Vision Instruction**: Excellent accuracy without any fine-tuning.
- **Molmo-7B-0-0924 VisionLM**: Outstanding results without fine-tuning, executable on GPUs with 16 GB of memory thanks to `bfloat16`.
- **LLaVA-1.6-Mistral**: High accuracy without fine-tuning, demonstrating computational efficiency.
- **GPT-4o and GPT-4o-mini**: The most performant models in their field, surpassing even YOLO, without the need for specific hardware thanks to the use of OpenAI APIs.

## Contact

For questions or collaboration, feel free to contact me:

**Pasquale Molinaro**

📩 Email: [pasqualemolinaro97@gmail.com](mailto:pasqualemolinaro97@gmail.com)  
🔗 LinkedIn: [Pasquale Molinaro](https://www.linkedin.com/in/pasquale-molinaro-8654021aa/)

---

Feel free to modify this `README.md` by adding specific details or updates related to your project!
