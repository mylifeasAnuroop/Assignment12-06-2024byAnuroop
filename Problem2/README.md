# Sales Conversation Dataset Generation using GPT-2

### Author:Anuroop Arya
## Abstract

This project aims to generate a sales conversation dataset using GPT-2, a state-of-the-art language model. The dataset is designed to simulate coherent and contextually appropriate conversations between a salesperson and a customer. This README provides an overview of the dataset generation process, including model setup, conversation initiation, and dataset storage.

## Table of Contents

1. [Introduction](#introduction)
2. [Project Overview](#project-overview)
3. [Setup](#setup)
4. [Model](#model)
5. [Dataset Generation](#dataset-generation)
6. [Instructions](#instructions)
7. [Conclusion](#conclusion)
8. [References](#references)

---

## Introduction

In the realm of language models and generative AI, generating coherent and engaging conversations is a significant challenge. This project focuses on using GPT-2 to create a sales conversation dataset that meets specific criteria for context relevance, coherence, creativity, and ethical considerations.

## Project Overview

### Problem Statement

The objective is to create a sales conversation dataset that includes a minimum of 100 sets of dialogues, with each response comprising 50-75 words. Conversations should be contextually relevant, coherent, engaging, and free from toxic language or bias. The dataset aims to simulate real-world sales interactions using AI-generated text.

### Performance Evaluation Criteria and Grading Rubric

- **Minimum Data Size Requirement**: Each submission must include at least 100 sets of dialogues.
- **Data Quality**:
  - **Contextual Relevance and Understanding**: Conversations should demonstrate a deep understanding of the product or service being sold.
  - **Coherence, Fluency, and Readability**: Texts must flow smoothly, be grammatically correct, and adhere to principles of effective communication.
  - **Creativity and Engagement**: Dialogues should be creative and engaging to capture the interest of prospective clients.
  - **Toxicity and Bias Mitigation**: Conversations must avoid toxic language, personal attacks, bias, and stereotypes.
  - **Accuracy and Completeness of Information**: The information presented should be accurate and complete.
  - **Compute Time**: The time taken to generate the data should be minimized.

### Bonus for Data Size Increment

Participants are incentivized to exceed the minimum data size requirement. Additional bonus points are awarded for every 10 sets of dialogues beyond the minimum.

### Submission Format

- **Code**: PyTorch code for model definition, training, and evaluation, along with code for generating the interactive visualization interface.
- **Documentation**: README.md file explaining the project details, model architecture, training process, and results. Instructions on using the visualization interface.
- **Visualization**: Link or notebook demonstrating the interactive visualization.

## Setup

### Environment

- **Language Model**: GPT-2
- **Deep Learning Framework**: PyTorch
- **Tokenizer**: GPT2Tokenizer
- **Model**: GPT2LMHeadModel

### Requirements

- Python 3.x
- PyTorch
- Transformers library
- Pandas
- Matplotlib
- Seaborn

## Model

### Model Architecture

GPT-2 is a transformer-based language model that uses attention mechanisms to process and generate text. The model architecture includes self-attention layers, feed-forward networks, and layer normalization.

## Dataset Generation

### Implementation

The dataset is generated using the following steps:

1. **Model Setup**: Load the pre-trained GPT-2 model and tokenizer.
2. **Conversation Initiation**: Start a sales conversation with an initial prompt.
3. **Conversation Generation**: Alternate responses between a salesperson and a customer.
4. **Data Storage**: Save the generated conversation to a CSV file.

## Instructions

### Running the Code

1. **Setup Environment**: Install the required libraries using `pip install -r requirements.txt`.
2. **Run the Code**: Execute the code snippet provided in a Python environment.

### Dataset Usage

The generated dataset can be used for training AI models, evaluating sales techniques, and analyzing conversational patterns.

## Conclusion

This project demonstrates the application of GPT-2 for generating a sales conversation dataset. It highlights the importance of context, coherence, and engagement in generating AI-driven sales dialogues.

## References

- Hugging Face Transformers Library: [https://huggingface.co/transformers/](https://huggingface.co/transformers/)
- PyTorch Documentation: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)

---

This README.md provides a comprehensive overview of the project, including setup instructions, model details, dataset generation process, and instructions for usage. Adjustments can be made to fit specific requirements or additional details as needed.
