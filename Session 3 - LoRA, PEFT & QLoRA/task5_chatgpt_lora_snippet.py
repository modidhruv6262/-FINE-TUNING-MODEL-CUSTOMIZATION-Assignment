from pathlib import Path
from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType

def run_chatgpt_lora_snippet():
    print("=" * 70)
    print("TASK 5: EXECUTING & HARDENING CHATGPT-GENERATED LORA SNIPPET")
    print("=" * 70)

    model_name = "distilbert-base-uncased"
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"],
        bias="none"
    )

    peft_model = get_peft_model(base_model, peft_config)
    print("ChatGPT LoRA Snippet successfully initialized and adapted!")
    peft_model.print_trainable_parameters()
    print("=" * 70)

if __name__ == "__main__":
    run_chatgpt_lora_snippet()
