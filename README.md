# Generative AI for PPE (Personal Protective Equipment) Detection

This repository contains a collection of Generative Artificial Intelligence models used for detecting Personal Protective Equipment (PPE) in images. The project's goal is to compare different Vision Language Models (VLMs) and object detection models to evaluate their performance in the task of PPE detection, both with and without fine-tuning.

By leveraging state-of-the-art AI models, we aim to enhance workplace safety by accurately identifying whether individuals are wearing the required PPE. This is crucial in various industries such as construction, manufacturing, and healthcare, where compliance with safety regulations is essential.

## 📖 Table of Contents

- [Introduction](#introduction)
- [Models Used](#models-used)
  - [YOLOv8](#yolov8)
  - [Phi-3.5 Vision Instruction](#phi-35-vision-instruction)
  - [Molmo-7B-0-0924 VisionLM](#molmo-7b-0-0924-visionlm)
  - [LLaVA-1.6-Mistral](#llava-16-mistral)
  - [GPT-4o and GPT-4o-mini](#gpt-4o-and-gpt-4o-mini)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Conclusion](#conclusion)
- [Contact](#contact)

## Introduction

Automatic detection of PPE is vital for ensuring workplace safety and compliance with occupational health and safety regulations. Traditional methods of monitoring PPE compliance are often manual, time-consuming, and prone to human error.

With the advancements in Generative AI and computer vision, it is now possible to automate this process, providing real-time detection and alerts. This project explores the effectiveness of various state-of-the-art models in detecting PPE in images, aiming to identify the most efficient and accurate approaches.

## Models Used

In this project, we have utilized a combination of object detection models and Vision Language Models (VLMs). Below is an overview of each model, along with their performance and hardware requirements.

### YOLOv8

- **File**: `yolov8.ipynb`
- **Description**: YOLOv8 (You Only Look Once version 8) is an advanced real-time object detection model known for its speed and accuracy. We fine-tuned YOLOv8 on a specific PPE dataset to enhance its performance in detecting various PPE items such as helmets, safety vests, goggles, and gloves.
- **Performance**: Achieved an accuracy higher than 80% in the PPE detection task after fine-tuning.
- **Hardware Used**: NVIDIA GeForce RTX 3070 Ti GPU.
- **Advantages**:
  - Fast inference suitable for real-time applications.
  - High accuracy with fine-tuning.
- **Limitations**:
  - Requires significant computational resources for training.
  - Performance depends on the quality and size of the training dataset.

### Phi-3.5 Vision Instruction

- **File**: `phi-3.5-vision.ipynb`
- **Description**: Phi-3.5 Vision Instruction is a Vision Language Model developed by Google, containing 4.2 billion parameters. It integrates visual understanding with language processing, enabling it to interpret images and provide descriptive outputs.
- **Performance**: Achieves excellent accuracy without any fine-tuning, demonstrating strong generalization capabilities.
- **Hardware Used**: NVIDIA L4 GPU with 24 GB of memory.
- **Advantages**:
  - No fine-tuning required, saving time and resources.
  - Capable of understanding complex visual scenes.
- **Limitations**:
  - Large model size requires substantial memory.
  - Inference may be slower compared to smaller models.

### Molmo-7B-0-0924 VisionLM

- **File**: `molmo-7b-visionlm.ipynb`
- **Description**: Molmo-7B-0-0924 VisionLM is an optimized Vision Language Model with 7 billion parameters. It utilizes `bfloat16` precision to reduce memory consumption, making it executable on GPUs with limited memory.
- **Performance**: Obtains outstanding results without any fine-tuning, effectively detecting PPE items in images.
- **Hardware Used**: NVIDIA Tesla P100 GPU with 16 GB of memory.
- **Advantages**:
  - Memory-efficient due to `bfloat16` precision.
  - Strong performance without fine-tuning.
- **Limitations**:
  - May require specialized hardware supporting `bfloat16`.
  - Potentially less accurate than larger models like GPT-4.

### LLaVA-1.6-Mistral

- **File**: `llava-1.6-mistral.ipynb`
- **Description**: LLaVA-1.6-Mistral is a Vision Language Model that combines language understanding with visual processing. It is designed to be efficient, providing high accuracy without extensive computational resources.
- **Performance**: Demonstrates high accuracy in PPE detection without fine-tuning, and is computationally efficient.
- **Hardware Used**: NVIDIA Tesla P100 GPU with 16 GB of memory.
- **Advantages**:
  - Efficient in terms of computational resource usage.
  - Good performance out-of-the-box.
- **Limitations**:
  - May not capture all the nuances in complex images compared to larger models.

### GPT-4o and GPT-4o-mini

- **Files**: `gpt-4o.ipynb` and `gpt-4o-mini.ipynb`
- **Description**: GPT-4o and GPT-4o-mini are Vision Language Models leveraging OpenAI's GPT-4 architecture, known for its advanced language and visual understanding capabilities. They interface with the model via OpenAI's API.
- **Performance**: Achieve exceptional results in PPE detection, surpassing other models, including YOLO, without the need for specialized hardware.
- **Implementation**: Use HTTP requests to interact with OpenAI's API, processing images and receiving descriptive outputs.
- **Advantages**:
  - State-of-the-art performance in both language and visual tasks.
  - No need for local computational resources; computation is offloaded to OpenAI's servers.
- **Limitations**:
  - Requires an OpenAI API key with access to GPT-4.
  - Potential costs associated with API usage.
  - Latency depends on network connectivity.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)
- CUDA-compatible GPU (for local execution of models)
- OpenAI account with API access (for GPT-4o and GPT-4o-mini)
- Jupyter Notebook or JupyterLab

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/pasquale9197/Generative-AI.git
cd Generative-AI
```
## Set Up API Keys (For GPT-4 Models)
* Obtain an API key from OpenAI.
* Set the API key as an environment variable:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Results
The models have shown the following performance in the PPE detection task:

* YOLOv8:
* * Accuracy: Over 80% after fine-tuning.
* * Pros: Real-time detection capabilities.
* * Cons: Requires fine-tuning and significant computational resources.

* Phi-3.5 Vision Instruction:
* * Accuracy: High accuracy without fine-tuning.
* * Pros: Strong out-of-the-box performance.
* * Cons: Large model size requires substantial memory.

* Molmo-7B-0-0924 VisionLM:
* * Accuracy: Excellent without fine-tuning.
* * Pros: Memory-efficient execution.
* * Cons: May need hardware supporting bfloat16 precision.

* LLaVA-1.6-Mistral:
* * Accuracy: High accuracy without fine-tuning.
* * Pros: Computationally efficient.
* * Cons: May not perform as well on complex images.

* GPT-4o and GPT-4o-mini:
* * Accuracy: Exceptional, surpassing other models.
* * Pros: No need for local computational resources.
* * Cons: Requires OpenAI API access; potential costs.

## Conclusion
This project demonstrates that advanced Vision Language Models can effectively detect PPE in images without the need for extensive fine-tuning or specialized hardware. Models like GPT-4 show exceptional performance but come with considerations such as API access and usage costs.

For real-time applications where local execution is required, models like YOLOv8 offer fast and accurate detection but require significant computational resources for training and inference.

Selecting the appropriate model depends on the specific requirements, including accuracy needs, available hardware, and whether real-time processing is necessary.

Contact
For questions or collaboration, feel free to contact me:

Pasquale Molinaro

📩 Email: pasqualemolinaro97@gmail.com
🔗 LinkedIn: [Pasquale Molinaro](https://www.linkedin.com/in/pasquale-molinaro-8654021aa/)
