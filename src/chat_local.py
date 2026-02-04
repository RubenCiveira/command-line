"""
Chat local usando transformers, sin depender de Ollama.

Incluye una selección de modelos ligeros de HF Hub, todos libres
y sin requisitos de licencia.

Uso:
    python src/chat_local.py
    python src/chat_local.py --model Qwen/Qwen2.5-1.5B-Instruct
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from rich.console import Console
from rich.prompt import IntPrompt

console = Console()

MODELS = [
    ("Qwen/Qwen2.5-0.5B-Instruct", "0.5B", "~1 GB", "Mínimo viable, respuestas básicas"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "1.5B", "~3 GB", "Buen equilibrio tamaño/calidad"),
    ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "1.7B", "~3.4 GB", "Diseñado para ser compacto"),
    ("Qwen/Qwen2.5-3B-Instruct", "3B", "~6 GB", "Buena calidad general"),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "1.1B", "~2.2 GB", "Clásico ligero"),
    ("microsoft/Phi-3.5-mini-instruct", "3.8B", "~8 GB", "Mejor calidad, más pesado"),
]

DEFAULT_MODEL = MODELS[1][0]  # Qwen2.5-1.5B-Instruct


def detect_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def select_model() -> str:
    console.print("\n[bold]Modelos disponibles:[/bold]\n")
    for i, (name, params, ram, desc) in enumerate(MODELS, 1):
        console.print(f"  [{i}] {name}")
        console.print(f"      [dim]{params} params · {ram} float16 · {desc}[/dim]")
    console.print()

    idx = IntPrompt.ask("Selecciona modelo", default=2)
    if 1 <= idx <= len(MODELS):
        return MODELS[idx - 1][0]
    console.print("[red]Opción no válida, usando modelo por defecto.[/red]")
    return DEFAULT_MODEL


def load_model(model_name: str, device: str):
    console.print(f"\n[yellow]Cargando modelo:[/yellow] {model_name}")
    console.print(f"[yellow]Dispositivo:[/yellow] {device}")
    console.print("[dim]Primera ejecución descarga los pesos...[/dim]\n")

    dtype = torch.float16 if device != "cpu" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
    ).to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    console.print("[green]Modelo cargado.[/green]\n")
    return model, tokenizer


def chat(model, tokenizer, device: str):
    console.print("[bold]Chat local[/bold] — escribe tu mensaje (vacío o 'exit' para salir)\n")

    messages = []
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    while True:
        user_input = console.input("[cyan]Tú:[/cyan] ")
        if not user_input.strip() or user_input.strip().lower() == "exit":
            break

        messages.append({"role": "user", "content": user_input})

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        console.print("[green]Asistente:[/green] ", end="")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                streamer=streamer,
            )

        # Decodificar solo los tokens nuevos para el historial
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        assistant_reply = tokenizer.decode(new_tokens, skip_special_tokens=True)
        messages.append({"role": "assistant", "content": assistant_reply})

        console.print()

    console.print("[dim]Saliendo...[/dim]")


def main():
    parser = argparse.ArgumentParser(description="Chat local con transformers")
    parser.add_argument("--model", default=None, help="Modelo HF Hub (omitir para menú interactivo)")
    args = parser.parse_args()

    device = detect_device()

    if args.model:
        model_name = args.model
    else:
        model_name = select_model()

    model, tokenizer = load_model(model_name, device)
    chat(model, tokenizer, device)


if __name__ == "__main__":
    main()
