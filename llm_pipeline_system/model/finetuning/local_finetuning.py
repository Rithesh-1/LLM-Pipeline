from typing import Literal

from llm_pipeline_system.model.finetuning.finetune import finetune, inference, save_model


def run_local_finetuning(
    finetuning_type: Literal["sft", "dpo"],
    num_train_epochs: int,
    per_device_train_batch_size: int,
    learning_rate: float,
    dataset_huggingface_workspace: str,
    is_dummy: bool = False,
) -> None:
    # Define output directory for the model
    output_dir = "./local_finetuned_model"
    model_name = "unsloth/llama-3.1-8b-bnb-4bit" # A default model for local finetuning

    model, tokenizer = finetune(
        finetuning_type=finetuning_type,
        model_name=model_name,
        output_dir=output_dir,
        dataset_huggingface_workspace=dataset_huggingface_workspace,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        is_dummy=is_dummy,
    )

    # Perform inference with the finetuned model
    inference(model, tokenizer)

    # Save the finetuned model locally
    save_model(model, tokenizer, output_dir, push_to_hub=False)

    print(f"Local finetuning completed. Model saved to {output_dir}")
