
```mermaid
sequenceDiagram
    participant User
    participant Local_Machine
    participant Training_Pipeline
    participant ClearML
    participant Hugging_Face
    participant MLflow

    User->>Local_Machine: poetry poe run-training-pipeline
    Local_Machine->>Training_Pipeline: Start Local Training
    Training_Pipeline->>Hugging_Face: Pull Base Model
    Training_Pipeline->>Training_Pipeline: Load Training & Preference Datasets
    Training_Pipeline->>Local_Machine: Fine-tune LLM (SFT or DPO)
    Local_Machine-->>Training_Pipeline: Training Metrics
    Training_Pipeline->>MLflow: Log Metrics
    Training_Pipeline->>ClearML: Log Experiment Details
    Training_Pipeline-->>Hugging_Face: Push Fine-tuned Model
    Training_Pipeline-->>User: Training Complete
```

**Explanation:**

1.  **Initiation:** The user initiates the training process by running the `poetry poe run-training-pipeline` command on their local machine.
2.  **Pipeline Start:** The command triggers the training pipeline.
3.  **Model Loading:** The pipeline pulls the base LLM from the Hugging Face model registry.
4.  **Data Loading:** It loads the instruction and preference datasets that were generated in the data processing stage.
5.  **Local Fine-Tuning:** The fine-tuning process for the LLM (either Supervised Fine-Tuning (SFT) or Direct Preference Optimization (DPO), depending on the configuration) is executed directly on the local machine.
6.  **Metrics Logging:** During and after training, training metrics are logged to MLflow for experiment tracking. The entire experiment, including code, dependencies, and logs, is also captured by ClearML.
7.  **Model Pushing:** Once training is complete, the fine-tuned model can be pushed to the Hugging Face model registry.
8.  **Completion:** The user is notified that the training process is complete.
