# onnx_greedy.py
import numpy as np
def greedy_generate_onnx(
    encoder_session,
    decoder_session,
    tokenizer,
    input_text,
    max_new_tokens=50,
    decoder_start_token_id=None,
):
    inputs = tokenizer(input_text, return_tensors="np")
    input_ids = inputs["input_ids"].astype(np.int64)
    attention_mask = inputs["attention_mask"].astype(np.int64)

    # Энкодер
    encoder_outputs = encoder_session.run(None,{
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
    )
    encoder_hidden_states = encoder_outputs[0]# До этого этапа очень понятно

    decoder_start_token_id = 2
    

    decoder_input_ids = np.array([[decoder_start_token_id]], dtype=np.int64)

    for _ in range(max_new_tokens):
        decoder_inputs = {
            "input_ids": decoder_input_ids,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": attention_mask,  # вот этот ключ добавили
        }

        decoder_outputs = decoder_session.run(None, decoder_inputs)
        logits = decoder_outputs[0]
        next_token_logits = logits[:, -1, :]
        next_token_id = np.argmax(next_token_logits, axis=-1).astype(np.int64)

        decoder_input_ids = np.concatenate([decoder_input_ids, next_token_id[:, None]], axis=1)

        if next_token_id[0] == tokenizer.eos_token_id:
            break

    return decoder_input_ids

