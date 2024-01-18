import argparse
import json
import uuid
from typing import Union

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from llama_cpp import Llama
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizerFast,
)

from functionary.inference_stream import generate_stream
from functionary.openai_types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatInput,
    ChatMessage,
    Choice,
    StreamChoice,
)
from functionary.prompt_template import get_prompt_template_from_tokenizer

app = FastAPI(title="Functionary API")

model = Llama(
    model_path="/workspace/functionary-medium-v2.2.q4_0.gguf",
    n_ctx=0,
    n_keep=-1,
    seed=0,
    n_gpu_layers=1,
    n_threads=3,
)
tokenizer = AutoTokenizer.from_pretrained(
    "meetkai/functionary-medium-v2.2", legacy=True, use_fast=False
)
# model = Llama(model_path="/Users/bijon/Desktop/AssistantsAPI-main/models/functionary-7b-v2.q4_0.gguf", n_ctx=8192, n_gpu_layers=1)
# tokenizer = AutoTokenizer.from_pretrained("meetkai/functionary-7b-v2", legacy=True, use_fast=False)


def get_response(messages, tools, temperature):
    prompt_template = get_prompt_template_from_tokenizer(tokenizer)

    # Before inference, we need to add an empty assistant (message without content or function_call)
    messages.append({"role": "assistant"})

    # Create the prompt to use for inference
    prompt_str = prompt_template.get_prompt_from_messages(messages, tools)
    token_ids = tokenizer.encode(prompt_str)

    gen_tokens = []
    # Get list of stop_tokens
    stop_token_ids = [
        tokenizer.encode(token)[-1]
        for token in prompt_template.get_stop_tokens_for_generation()
    ]
    print("stop_token_ids: ", stop_token_ids)

    # We use function generate (instead of __call__) so we can pass in list of token_ids
    for token_id in model.generate(token_ids, temp=temperature):
        if token_id in stop_token_ids:
            break
        gen_tokens.append(token_id)

    llm_output = tokenizer.decode(gen_tokens)

    # parse the message from llm_output
    result = prompt_template.parse_assistant_response(llm_output)
    return ChatMessage(**result)


@app.post("/v1/chat/completions")
async def chat_endpoint(chat_input: ChatInput):
    global model, tokenizer
    request_id = str(uuid.uuid4())
    if not chat_input.stream:
        response_message = get_response(
            messages=chat_input.messages,
            tools=chat_input.tools,
            temperature=chat_input.temperature,
        )
        finish_reason = "stop"
        if (
            response_message.function_call
            or response_message.tool_calls
            and len(response_message.tool_calls) > 0
        ):
            finish_reason = "tool_calls"  # need to add this to follow the format of openAI function calling
        result = ChatCompletion(
            id=request_id,
            choices=[Choice.from_message(response_message, finish_reason)],
        )
        return result
    else:
        response_generator = generate_stream(
            messages=chat_input.messages,
            functions=chat_input.functions,
            tools=chat_input.tools,
            temperature=chat_input.temperature,
            model=model,  # type: ignore
            tokenizer=tokenizer,
        )

        def get_response_stream():
            for response in response_generator:
                chunk = StreamChoice(**response)
                result = ChatCompletionChunk(id=request_id, choices=[chunk])
                chunk_dic = result.dict(exclude_unset=True)
                chunk_data = json.dumps(chunk_dic, ensure_ascii=False)
                yield f"data: {chunk_data}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(get_response_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Functionary API Server")
    parser.add_argument(
        "--model",
        type=str,
        default="musabgultekin/functionary-7b-v1",
        help="Model name",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="choose which device to host the model: cpu, cuda, cuda:xxx, or auto",
    )
    parser.add_argument("--load_in_8bit", type=bool, default=False)
    args = parser.parse_args()

    uvicorn.run(app="main:app", host="0.0.0.0", port=8080, reload=True)
