# Bug Detection and Fixing Project


##  Overview
This project focuses on **automated detection and fixing of software bugs**  deep learning models. By leveraging **pretrained transformer-based models** such as **CodeBERT** and **DeepSeek-Coder**, the system can intelligently identify errors in Python source code and suggest appropriate corrections—without requiring any manual intervention.
The goal is to build an end-to-end AI-powered pipeline that:
- **Detects bugs** in Python code snippets,
- **Predicts the cause or type of error**, and
- **Automatically fixes the buggy code** using generative models.
This system aims to assist developers in debugging tasks, enhance code quality, and potentially integrate with IDEs for real-time suggestions.

## Dataset
The dataset used for training and testing consists of buggy Python code samples with labeled bug information.

**Training Dataset**: codenetpy_train.json

**Testing Dataset**: codenetpy_test.json
Each dataset entry includes:

- **original_src** : Buggy Python source code

- **changed_src** : Corrected Python source code

- **problem_id** : Unique problem identifier

- **error_class** : Type of error
## Setup and Installation
1. ### Clone the repository:```sh git clone https://github.com/Shreyanshy53/BugDetectionProject.git cd BugDetectionProject

2. Install dependencies: pip install torch transformers tqdm gradio autopep8
  
3. Ensure that the dataset is placed in the appropriate directory -> ``` (/content/drive/MyDrive/BugDetectionProject/) ```
## Approach
The project follows these key steps:
- **Preprocessing:** This step involves cleaning the dataset by removing unnecessary characters, tokenizing the source code into meaningful units, and formatting it to ensure consistency for model input.
- **Feature Extraction**: Using Abstract Syntax Trees (AST) and token-based analysis.
- **Data Pipeline Implementation**: Organizing data processing workflows to handle large datasets efficiently.
- **Bug Detection**: Identifying errors in the code using pretrained models (`CodeBERT` and `DeepSeek-AI/DeepSeek-Coder-1.3B-Instruct`).
- **Automatic Bug Fixing**: Generating corrected versions of the buggy code using `DeepSeek-AI/DeepSeek-Coder-1.3B-Instruct`.

## Data Pipeline

A structured data pipeline was implemented to streamline preprocessing and feature extraction:

- **Loading Data**: Parsing JSON datasets efficiently.
- **Preprocessing**: Cleaning and normalizing code.
- **Tokenization**: Converting code into tokens for model input.
- **AST Analysis**: Extracting syntax structure of code.
- **Batch Processing**: Handling large datasets in batches for memory efficiency.
- **Dataset Preparation**: Converting processed data into PyTorch datasets.
## Bug Detection and Fixing using Pretrained Models
The project uses **CodeBERT** and **DeepSeek** for bug detection and **DeepSeek-AI/DeepSeek-Coder-1.3B-Instruct** for bug fixing without fine-tuning.
```sh from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

def detect_bug_and_fix(code_input):
    bug_instruction = f"### Bug Detection:\n{code_input}\n\n### Error Message:\n"
    bug_inputs = tokenizer(bug_instruction, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad():
        bug_outputs = model.generate(**bug_inputs, max_length=128, do_sample=True, temperature=0.6, top_p=0.8)

    error_message = tokenizer.decode(bug_outputs[0], skip_special_tokens=True)
    error_message = error_message.split("### Error Message:")[-1].strip()

    fix_instruction = f"### Buggy Code:\n{code_input}\n\n### Fixed Code:\n"
    fix_inputs = tokenizer(fix_instruction, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad():
        fix_outputs = model.generate(**fix_inputs, max_length=256, do_sample=True, temperature=0.6, top_p=0.8)

    fixed_code = tokenizer.decode(fix_outputs[0], skip_special_tokens=True)
    fixed_code = fixed_code.split("### Fixed Code:")[-1].strip()

    return error_message, fixed_code

```


## Architecture Diagram
   ![Flowchart](https://github.com/Shreyanshy53/Bug_DetectionFixing/blob/main/flowchart.jpg?raw=true) 
##  Future Plans
- **Enhancing Accuracy** : Experimenting with additional pretrained models.

- **Expanding Language Support** : Extending the model to support multiple programming languages.

- **Real-time Bug Detection** : Integrating with IDEs for live bug detection and fixes.

- **Interactive Debugging UI** : Developing a web-based interface for debugging assistance.

- **Model Optimization** : Reducing computational costs for faster performance.

- **Advanced Data Pipeline** : Automating preprocessing, feature extraction, and batch processing for scalability.

##  Conclusion
This project provides a foundation for using deep learning models to detect and automatically fix software bugs without fine-tuning. Future improvements can involve integrating real-time debugging tools and experimenting with multiple models for better performance.




