from huggingface_hub import InferenceClient
import gradio as gr
import os

client = InferenceClient(
    "mistralai/Mistral-7B-Instruct-v0.2",
    token=os.environ.get("HF_TOKEN")
    provider="together"
)


def format_prompt(message, history):
    messages = []
    for user_prompt, bot_response in history:
        messages.append({"role": "user", "content": user_prompt})
        messages.append({"role": "assistant", "content": bot_response})
    messages.append({"role": "user", "content": message})
    return messages

def generate(
    prompt, history, temperature=0.9, max_new_tokens=256, top_p=0.95, repetition_penalty=1.0,
):
    temperature = float(temperature)
    if temperature < 1e-2:
        temperature = 1e-2
    top_p = float(top_p)

    generate_kwargs = dict(
        temperature=temperature,
        max_tokens=max_new_tokens,
        top_p=top_p,
        seed=42,
    )

    # Get the list of message dictionaries
    messages_for_api = format_prompt(prompt, history)

    # Pass the messages list to the create method
    stream = client.chat.completions.create(
        messages=messages_for_api,  # This is the crucial change
        **generate_kwargs,
        stream=True,
    )
    output = ""

    for chunk in stream:
        # Check if the chunk contains choices and message content
        if chunk.choices and chunk.choices[0].delta.content:
            output += chunk.choices[0].delta.content
            yield output
    return output


additional_inputs=[
    gr.Slider(
        label="Temperature",
        value=0.9,
        minimum=0.0,
        maximum=1.0,
        step=0.05,
        interactive=True,
        info="Higher values produce more diverse outputs",
    ),
    gr.Slider(
        label="Max new tokens",
        value=256,
        minimum=0,
        maximum=1048,
        step=64,
        interactive=True,
        info="The maximum numbers of new tokens",
    ),
    gr.Slider(
        label="Top-p (nucleus sampling)",
        value=0.90,
        minimum=0.0,
        maximum=1,
        step=0.05,
        interactive=True,
        info="Higher values sample more low-probability tokens",
    ),
    gr.Slider(
        label="Repetition penalty",
        value=1.2,
        minimum=1.0,
        maximum=2.0,
        step=0.05,
        interactive=True,
        info="Penalize repeated tokens",
    )
]

css = """
  #mkd {
    height: 500px;
    overflow: auto;
    border: 1px solid #ccc;
  }
"""

with gr.Blocks(css=css) as demo:
    gr.HTML("<h1><center>Mistral 7B Instruct<h1><center>")
    gr.HTML("<h3><center>In this demo, you can chat with <a href='https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1'>Mistral-7B-Instruct</a> model. 💬<h3><center>")
    gr.HTML("<h3><center>Learn more about the model <a href='https://huggingface.co/docs/transformers/main/model_doc/mistral'>here</a>. 📚<h3><center>")
    gr.ChatInterface(
        generate,
        additional_inputs=additional_inputs,
        examples=[["What is the secret to life?"], ["Write me a recipe for pancakes."]]
    )

demo.queue(default_concurrency_limit=75, max_size=100).launch(debug=True)
