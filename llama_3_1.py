!pip install transformers
!pip install torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4" 
import transformers
import torch



def generate_response(
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
        token="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",
    )

    try:
        print("In prompt")
        prompt = "What is the meaning of life"
        max_tokens = 150
        if not max_tokens:
            max_tokens = max_gen_len
        messages = [
            {"role": "system", "content": "You are a text summarizing tool."},
            {"role": "user", "content": prompt},
        ]
        print("there I am")
        outputs = pipeline(
            messages,
            max_new_tokens=256,
            temperature=temperature,
            top_p=top_p,
            max_seq_len = max_seq_len,
            max_batch_size = max_batch_size,
        )            
        print("model loaded")
        print(f'LLaMA raw result: {outputs}')
        print(f'LLaMA Result: {outputs[0]["generation"]["content"]}')
        return outputs[0]['generation']['content']
    except Exception as ex:
        print(f'Exception: {ex} while processing task')
        return f'Could not generate text, Exception: {ex.__class__}'

print(generate_response())


