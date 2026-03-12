import atexit
import html
import signal
import time
from typing import Any, Dict, List, Optional

import gradio as gr
from openai import OpenAI

from config import AVAILABLE_MODELS, VLLM_HOST, VLLM_PORT, VLLM_API_KEY
from model_manager import VLLMManager

manager = VLLMManager()


def cleanup_and_exit(signum=None, frame=None):
    try:
        manager.stop_server()
    finally:
        if signum is not None:
            raise SystemExit(0)


atexit.register(manager.stop_server)
signal.signal(signal.SIGINT, cleanup_and_exit)
signal.signal(signal.SIGTERM, cleanup_and_exit)


def get_openai_client():
    return OpenAI(
        base_url=f"http://{VLLM_HOST}:{VLLM_PORT}/v1",
        api_key=VLLM_API_KEY,
    )


# --------------------------------------------------
# GPU helpers with stable values
# --------------------------------------------------

def _parse_gpu_indices(selected_values: Optional[List[str]]) -> Optional[List[int]]:
    """Convertit ['0','3'] -> [0,3]."""
    if not selected_values:
        return None
    result = []
    for v in selected_values:
        try:
            result.append(int(v))
        except Exception:
            continue
    return sorted(result) if result else None


def _get_all_gpu_choices() -> List[tuple[str, str]]:
    """
    Retourne des choix Gradio stables:
    [(label dynamique, value stable), ...]
    """
    gpus = manager.get_all_gpu_list()
    return [(g["label"], str(g["index"])) for g in gpus]


def _remap_gpu_selection(old_selected: Optional[List[str]], new_choices: List[tuple[str, str]]) -> List[str]:
    """
    Conserve uniquement les values stables encore présentes dans les nouveaux choix.
    Si rien n'est conservé, tout sélectionner par défaut.
    """
    available_values = {value for _, value in new_choices}
    kept = [v for v in (old_selected or []) if v in available_values]
    if kept:
        return kept
    return [value for _, value in new_choices]


# --------------------------------------------------
# UI helpers
# --------------------------------------------------

def help_label(title: str, tooltip: str) -> str:
    return f"""
    <div class="help-label">
        <span class="help-label-text">{html.escape(title)}</span>
        <span class="help-icon" tabindex="0">
            ?
            <span class="help-popup">{html.escape(tooltip)}</span>
        </span>
    </div>
    """


# --------------------------------------------------
# Benchmark table
# --------------------------------------------------

def _build_benchmark_row(
    selected_label: str,
    enable_aiter: bool,
    tp_size: str,
    temperature: float,
    max_tokens: int,
    metrics: Dict[str, Any],
    vram_snapshot: Dict[str, Any],
    enforce_eager: bool = True,
    gpu_memory_utilization: float = 0.5,
) -> Dict[str, Any]:
    return {
        "model_label": selected_label,
        "aiter": "on" if enable_aiter else "off",
        "tp": int(tp_size),
        "eager": "yes" if enforce_eager else "no",
        "gpu_util": gpu_memory_utilization,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "ttft_ms": round(float(metrics.get("ttft_ms", 0.0)), 2),
        "throughput_tps": round(float(metrics.get("throughput_tps", 0.0)), 2),
        "e2e_latency_s": round(float(metrics.get("e2e_latency_s", 0.0)), 3),
        "prompt_tokens": int(metrics.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(metrics.get("completion_tokens", 0) or 0),
        "total_tokens": int(metrics.get("total_tokens", 0) or 0),
        "per_gpu_used_gb": str(vram_snapshot.get("per_gpu_used_gb", "")),
    }


_BENCH_HEADERS = [
    ("Model",               "Logical model selector. The displayed label maps to a concrete Hugging Face model ID used by the vLLM OpenAI-compatible server."),
    ("AITER",               "Enables ROCm AITER kernels (VLLM_ROCM_USE_AITER=1). Can replace default kernels with optimized ROCm implementations for attention, GEMM, MoE and normalization paths."),
    ("TP",                  "Number of shards used for tensor parallelism. Should not exceed the number of visible GPUs."),
    ("Eager",               "Disables CUDA/HIP graph compilation (--enforce-eager). Strongly recommended for TP>1 on ROCm: avoids compilation timeouts and speeds up startup. Slight runtime performance loss but much more stable."),
    ("GPU util",            "Fraction of GPU VRAM reserved for vLLM (model weights + KV cache). A lower value leaves more memory free but reduces KV cache size."),
    ("Temperature",         "Softmax temperature applied during sampling. Higher values flatten the output distribution; lower values make decoding more deterministic."),
    ("Max tokens",          "Upper bound on generated completion length. Directly affects decode time, KV-cache growth and end-to-end latency."),
    ("TTFT (ms)",           "Time To First Token: wall-clock delay between request submission and arrival of the first decoded token. Captures prefill cost and scheduling overhead."),
    ("Throughput (tok/s)",  "Generation throughput: completion_tokens / decode_duration (first to last token). Isolates steady-state decode performance from prefill."),
    ("E2E latency (s)",     "Total wall-clock latency from request dispatch to completion of the full streamed response, including prefill, decode and bookkeeping."),
    ("Prompt tokens",       "Number of tokens in the prompt after chat template expansion and tokenization. Strongly influences prefill latency and KV-cache footprint."),
    ("Completion tokens",   "Number of output tokens generated for the current request. Used with decode duration to compute generation throughput."),
    ("Total tokens",        "Sum of prompt tokens and completion tokens reported by the OpenAI-compatible response usage object."),
    ("Per-GPU VRAM",        "Per-device VRAM usage snapshot at generation time. Helps identify imbalance across GPUs and observe the impact of tensor parallelism."),
]


def _bench_header_cell(label: str, tip: str) -> str:
    escaped_tip = html.escape(tip, quote=True)
    icon = f'<span class="help-icon help-icon-th" data-tip="{escaped_tip}" tabindex="0">?</span>'
    return f"<th>{html.escape(label)}{icon}</th>"


def _render_benchmark_html(history_rows: List[Dict[str, Any]]) -> str:
    head = "".join(_bench_header_cell(label, tip) for label, tip in _BENCH_HEADERS)
    if not history_rows:
        body = f'<tr><td colspan="{len(_BENCH_HEADERS)}" style="text-align:center;color:#6b7280;padding:12px">No data yet — send a message to record a run.</td></tr>'
    else:
        body = ""
        for row in history_rows:
            cells = [
                row["model_label"],
                row["aiter"],
                row["tp"],
                row["eager"],
                row["gpu_util"],
                row["temperature"],
                row["max_tokens"],
                row["ttft_ms"],
                row["throughput_tps"],
                row["e2e_latency_s"],
                row["prompt_tokens"],
                row["completion_tokens"],
                row["total_tokens"],
                row["per_gpu_used_gb"],
            ]
            row_cells = "".join(
                f'<td class="vram-cell">{html.escape(str(c))}</td>' if i == len(cells) - 1
                else f"<td>{html.escape(str(c))}</td>"
                for i, c in enumerate(cells)
            )
            body += "<tr>" + row_cells + "</tr>"
    return f'<div id="benchmark_html_table"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


# --------------------------------------------------
# Model actions
# --------------------------------------------------

def load_model(
    selected_label: str,
    enable_aiter: bool,
    tp_size: str,
    selected_gpu_values: List[str],
    gpu_memory_utilization: float,
    enforce_eager: bool,
):
    model_id = AVAILABLE_MODELS[selected_label]
    selected_gpu_indices = _parse_gpu_indices(selected_gpu_values)
    try:
        manager.start_server_async(
            model_name=model_id,
            enable_aiter=enable_aiter,
            tensor_parallel_size=int(tp_size),
            selected_gpu_indices=selected_gpu_indices,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
        )
        status = manager.get_status()
        return status, gr.update(interactive=False)
    except Exception as e:
        return f"❌ Failed to launch vLLM: {e}", gr.update(interactive=False)


def unload_model():
    try:
        manager.stop_server()
        status = manager.get_status()
        return status, gr.update(interactive=False)
    except Exception as e:
        return f"❌ Failed to unload vLLM: {e}", gr.update(interactive=False)


# --------------------------------------------------
# Refresh
# --------------------------------------------------

def refresh_status_and_logs(selected_gpu_values: Optional[List[str]] = None):
    status = manager.get_status()
    ready = manager.is_model_loaded()
    new_choices = _get_all_gpu_choices()
    new_selected = _remap_gpu_selection(selected_gpu_values or [], new_choices)
    return status, gr.update(interactive=ready), gr.update(choices=new_choices, value=new_selected)


def refresh_gpu_controls():
    all_choices = _get_all_gpu_choices()
    info = manager.get_runtime_info()
    tp_choices = [str(x) for x in info["tp_choices"]]
    return (
        gr.update(choices=all_choices, value=[value for _, value in all_choices]),
        gr.update(choices=tp_choices, value=str(info["default_tp"])),
    )


def update_tp_from_gpu_selection(selected_gpu_values: List[str]):
    n = max(1, len(selected_gpu_values)) if selected_gpu_values else 1
    tp_choices = [str(x) for x in range(1, n + 1)]
    return gr.update(choices=tp_choices, value="1")


def clear_benchmark_history():
    return [], _render_benchmark_html([])


def clear_chat_history():
    return []


# --------------------------------------------------
# Chat
# --------------------------------------------------

def chat_fn(
    message: str,
    chat_history: List[Dict[str, str]],
    benchmark_history: List[Dict[str, Any]],
    selected_label: str,
    temperature: float,
    max_tokens: int,
    enable_aiter: bool,
    tp_size: str,
    selected_gpu_values: List[str],
    enforce_eager: bool,
    gpu_memory_utilization: float,
):
    if not message or not message.strip():
        return (
            chat_history,
            chat_history,
            benchmark_history,
            _render_benchmark_html(benchmark_history),
            "",
        )

    model_id = AVAILABLE_MODELS[selected_label]

    if not manager.is_model_loaded():
        error_text = "The selected model is not ready yet. Please load it first and wait until vLLM is fully ready."
        updated_chat = list(chat_history) + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": error_text},
        ]
        return (
            updated_chat,
            updated_chat,
            benchmark_history,
            _render_benchmark_html(benchmark_history),
            "",
        )

    if manager.current_model != model_id:
        error_text = (
            f"A different model is currently loaded in vLLM.\n"
            f"Selected in UI: {model_id}\n"
            f"Currently loaded: {manager.current_model}\n"
            f"Reload the selected model, or switch the dropdown to the loaded one."
        )
        updated_chat = list(chat_history) + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": error_text},
        ]
        return (
            updated_chat,
            updated_chat,
            benchmark_history,
            _render_benchmark_html(benchmark_history),
            "",
        )

    runtime_cfg = manager.current_runtime_config()
    expected_cfg = {
        "enable_aiter": enable_aiter,
        "tensor_parallel_size": int(tp_size),
    }
    if runtime_cfg != expected_cfg:
        mismatch_text = (
            "The active vLLM server configuration does not match the UI options. "
            "Reload the model to apply the selected AITER/TP settings."
        )
        updated_chat = list(chat_history) + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": mismatch_text},
        ]
        return (
            updated_chat,
            updated_chat,
            benchmark_history,
            _render_benchmark_html(benchmark_history),
            "",
        )

    messages = []
    for msg_item in chat_history:
        role = msg_item.get("role")
        content = msg_item.get("content")
        if role in {"user", "assistant", "system"} and content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": message})

    client = get_openai_client()

    start_time = time.perf_counter()
    first_token_time = None
    answer_chunks: List[str] = []
    usage = None

    try:
        stream = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )

        for chunk in stream:
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                usage = chunk_usage

            if not getattr(chunk, "choices", None):
                continue

            delta = chunk.choices[0].delta
            token_text = getattr(delta, "content", None)
            if token_text:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                answer_chunks.append(token_text)

    except Exception as e:
        error_text = f"Generation failed: {e}"
        updated_chat = list(chat_history) + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": error_text},
        ]
        return (
            updated_chat,
            updated_chat,
            benchmark_history,
            _render_benchmark_html(benchmark_history),
            "",
        )

    end_time = time.perf_counter()
    answer = "".join(answer_chunks).strip()

    if not answer:
        answer = "(Empty response)"

    if first_token_time is None:
        first_token_time = end_time

    prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage is not None else 0
    completion_tokens = getattr(usage, "completion_tokens", 0) if usage is not None else 0
    total_tokens = getattr(usage, "total_tokens", 0) if usage is not None else 0

    decode_duration = max(end_time - first_token_time, 1e-9)
    completion_tokens_value = int(completion_tokens or 0)

    metrics = {
        "ttft_ms": (first_token_time - start_time) * 1000.0,
        "throughput_tps": completion_tokens_value / decode_duration if completion_tokens_value > 0 else 0.0,
        "e2e_latency_s": end_time - start_time,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }

    vram_snapshot = manager.get_gpu_memory_snapshot_for_benchmark(
        gpu_indices=_parse_gpu_indices(selected_gpu_values)
    )

    updated_chat = list(chat_history) + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]

    updated_benchmark_history = list(benchmark_history)
    updated_benchmark_history.append(
        _build_benchmark_row(
            selected_label,
            enable_aiter,
            tp_size,
            temperature,
            max_tokens,
            metrics,
            vram_snapshot,
            enforce_eager=enforce_eager,
            gpu_memory_utilization=gpu_memory_utilization,
        )
    )

    return (
        updated_chat,
        updated_chat,
        updated_benchmark_history,
        _render_benchmark_html(updated_benchmark_history),
        "",
    )


# --------------------------------------------------
# UI
# --------------------------------------------------

with gr.Blocks(title="Chat vLLM ROCm") as demo:
    gr.Markdown("# Chat local vLLM + Gradio + ROCm")

    chat_state = gr.State([])
    benchmark_state = gr.State([])

    with gr.Row():
        with gr.Column(scale=2):
            gr.HTML(help_label(
                "Model",
                "Logical model selector. The displayed label maps to a concrete Hugging Face model ID used by the vLLM OpenAI-compatible server."
            ))
            model_dropdown = gr.Dropdown(
                choices=list(AVAILABLE_MODELS.keys()),
                value=list(AVAILABLE_MODELS.keys())[0],
                show_label=False,
            )
        with gr.Column(scale=2):
            load_button = gr.Button("Load / Download model", variant="primary")
            unload_button = gr.Button("Unload model", variant="stop", elem_id="unload_button")

    with gr.Row():
        with gr.Column(scale=2):
            gr.HTML(help_label(
                "Visible GPUs",
                "Select the GPUs to use for vLLM inference. All are checked by default. The selection automatically updates the Tensor Parallel Size choices and VRAM metrics."
            ))
            _initial_gpu_choices = _get_all_gpu_choices()
            gpu_selector = gr.CheckboxGroup(
                choices=_initial_gpu_choices,
                value=[value for _, value in _initial_gpu_choices],
                show_label=False,
                elem_id="gpu_selector",
            )
        with gr.Column(scale=1):
            gr.HTML(help_label(
                "Tensor parallel size",
                "Number of shards used for tensor parallelism. vLLM partitions eligible weight tensors across TP ranks; the selected value should not exceed the number of visible GPUs on this node."
            ))
            tp_dropdown = gr.Dropdown(
                choices=["1"],
                value="1",
                show_label=False,
            )

        with gr.Column(scale=1):
            gr.HTML(help_label(
                "Enable AITER",
                "Enables ROCm AITER kernels through the parent environment switch VLLM_ROCM_USE_AITER=1. This can replace selected default kernels with optimized ROCm implementations for attention, GEMM, MoE and normalization paths, depending on model, backend and installed vLLM build."
            ))
            enable_aiter = gr.Checkbox(
                value=False,
                show_label=False,
            )

        with gr.Column(scale=1):
            gr.HTML(help_label(
                "Enforce Eager",
                "Disables CUDA/HIP graph compilation (--enforce-eager). Strongly recommended for TP>1 on ROCm: avoids compilation timeouts and speeds up startup. Slight runtime performance loss but much more stable."
            ))
            enforce_eager = gr.Checkbox(
                value=False,
                show_label=False,
            )

    gr.HTML(help_label(
        "vLLM status / logs",
        "Combined process status and filtered vLLM logs. Health-check noise is suppressed so startup traces and runtime diagnostics remain readable."
    ))
    status_box = gr.Textbox(
        lines=10,
        max_lines=10,
        interactive=False,
        autoscroll=False,
        elem_id="status_box",
        show_label=False,
    )

    with gr.Row():
        with gr.Column():
            gr.HTML(help_label(
                "Temperature",
                "Softmax temperature applied during sampling. Higher values flatten the output distribution and increase entropy; lower values make decoding more deterministic."
            ))
            temperature = gr.Slider(
                0.0,
                1.0,
                value=0.7,
                step=0.1,
                show_label=False,
            )
        with gr.Column():
            gr.HTML(help_label(
                "Max new tokens",
                "Upper bound on generated completion length for the current request. This directly affects decode time, KV-cache growth and end-to-end latency."
            ))
            max_tokens = gr.Slider(
                32,
                4096,
                value=512,
                step=32,
                show_label=False,
            )
        with gr.Column():
            gr.HTML(help_label(
                "GPU memory utilization",
                "Fraction of GPU VRAM reserved for vLLM (model weights + KV cache). A lower value leaves more memory free for other processes but reduces the available KV cache size and maximum context length."
            ))
            gpu_mem_utilization = gr.Slider(
                0.0,
                1.0,
                value=0.5,
                step=0.01,
                show_label=False,
                elem_id="gpu_mem_utilization",
            )

    gr.Markdown(
        """
**Benchmark history**  
One row is appended after each completed generation so you can compare models and runtime settings across runs.
"""
    )

    benchmark_table = gr.HTML(value=_render_benchmark_html([]))

    with gr.Row():
        clear_bench_button = gr.Button("Clear benchmark history")
        clear_chat_button = gr.Button("Clear chat")

    with gr.Group(elem_id="chat_panel"):
        gr.HTML("<div class='section-title'>Chat</div>")
        chatbot = gr.Chatbot(label="Chat", height=450, elem_id="chatbot_box")

    with gr.Group(elem_id="prompt_panel"):
        gr.HTML("<div class='section-title'>Prompt</div>")
        msg = gr.Textbox(
            label="Prompt",
            placeholder="Type a prompt and press Enter…",
            lines=3,
            elem_id="prompt_box",
        )
        send_button = gr.Button("Send", variant="primary", elem_id="send_button", interactive=False)

    clear_chat_button.click(
        fn=clear_chat_history,
        inputs=[],
        outputs=[chat_state],
    ).then(
        fn=lambda: [],
        inputs=[],
        outputs=[chatbot],
    )

    load_button.click(
        fn=load_model,
        inputs=[model_dropdown, enable_aiter, tp_dropdown, gpu_selector, gpu_mem_utilization, enforce_eager],
        outputs=[status_box, send_button],
    )

    unload_button.click(
        fn=unload_model,
        inputs=[],
        outputs=[status_box, send_button],
    )

    demo.load(
        fn=refresh_gpu_controls,
        outputs=[gpu_selector, tp_dropdown],
        js="""() => {
        if (!document.getElementById('bench-floater')) {
            const floater = document.createElement('div');
            floater.id = 'bench-floater';
            document.body.appendChild(floater);

            let hideTimer = null;

            document.addEventListener('mousemove', function(e) {
                const icon = e.target.closest('.help-icon-th');
                if (icon) {
                    clearTimeout(hideTimer);
                    const text = (icon.getAttribute('data-tip') || '').trim();
                    if (!text) return;
                    floater.textContent = text;
                    floater.style.display = 'block';
                    const r = icon.getBoundingClientRect();
                    const fw = 320;
                    const spaceRight = window.innerWidth - r.right - 12;
                    const left = spaceRight >= fw ? r.right + 8 : Math.max(4, r.left - fw - 8);
                    floater.style.left = left + 'px';
                    floater.style.top = Math.max(4, r.bottom + 6) + 'px';
                } else {
                    clearTimeout(hideTimer);
                    hideTimer = setTimeout(function() { floater.style.display = 'none'; }, 80);
                }
            });
        }
    }""",
    )

    gpu_selector.change(
        fn=update_tp_from_gpu_selection,
        inputs=[gpu_selector],
        outputs=[tp_dropdown],
    )

    submit_inputs = [
        msg,
        chat_state,
        benchmark_state,
        model_dropdown,
        temperature,
        max_tokens,
        enable_aiter,
        tp_dropdown,
        gpu_selector,
        enforce_eager,
        gpu_mem_utilization,
    ]
    submit_outputs = [
        chatbot,
        chat_state,
        benchmark_state,
        benchmark_table,
        msg,
    ]

    send_button.click(
        fn=chat_fn,
        inputs=submit_inputs,
        outputs=submit_outputs,
        concurrency_limit=1,
    )
    msg.submit(
        fn=chat_fn,
        inputs=submit_inputs,
        outputs=submit_outputs,
        concurrency_limit=1,
    )

    clear_bench_button.click(
        fn=clear_benchmark_history,
        inputs=[],
        outputs=[benchmark_state, benchmark_table],
    )

    timer = gr.Timer(2.0)
    timer.tick(
        fn=refresh_status_and_logs,
        inputs=[gpu_selector],
        outputs=[status_box, send_button, gpu_selector],
    )


try:
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        css="""
/* Floating tooltip for benchmark table headers */
#bench-floater {
    display: none;
    position: fixed;
    z-index: 9999;
    background: #111827;
    color: white;
    border-radius: 10px;
    padding: 10px 12px;
    font-size: 12px;
    font-weight: 400;
    line-height: 1.4;
    max-width: 320px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.25);
    pointer-events: none;
}

.gradio-container {
    width: 90vw;
    max-width: 90vw;
    margin: 0 auto;
    padding-left: 8px;
    padding-right: 8px;
}

footer {
    display: none !important;
}

#status_box textarea {
    height: 220px !important;
    min-height: 220px !important;
    max-height: 220px !important;
    overflow-y: auto !important;
}

.help-label {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
    font-weight: 600;
    font-size: 14px;
}

.help-icon {
    position: relative;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 18px;
    height: 18px;
    border-radius: 999px;
    background: #e5e7eb;
    color: #111827;
    font-size: 12px;
    cursor: help;
    border: 1px solid #cbd5e1;
    flex-shrink: 0;
}

.help-popup {
    visibility: hidden;
    opacity: 0;
    transition: opacity 0.15s ease;
    position: absolute;
    left: 24px;
    top: -6px;
    width: 360px;
    z-index: 1000;
    background: #111827;
    color: white;
    text-align: left;
    border-radius: 10px;
    padding: 10px 12px;
    font-weight: 400;
    font-size: 12px;
    line-height: 1.4;
    box-shadow: 0 10px 25px rgba(0,0,0,0.25);
}

.help-icon:hover .help-popup,
.help-icon:focus .help-popup {
    visibility: visible;
    opacity: 1;
}

.section-title {
    font-weight: 700;
    font-size: 16px;
    margin-bottom: 8px;
}

/* CHAT PANEL */
#chat_panel,
#chat_panel > div,
#chat_panel .gr-group,
#chat_panel .gr-block {
    background: #eaf4ff !important;
    border-radius: 14px !important;
}

#chat_panel {
    border: 2px solid #93c5fd !important;
    padding: 12px !important;
    margin-top: 12px !important;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08) !important;
}

#chatbot_box {
    border: 1px solid #93c5fd !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    background: #f4f9ff !important;
}

/* PROMPT PANEL */
#prompt_panel,
#prompt_panel > div,
#prompt_panel .gr-group,
#prompt_panel .gr-block {
    background: #fff4e8 !important;
    border-radius: 14px !important;
}

#prompt_panel {
    border: 2px solid #fdba74 !important;
    padding: 12px !important;
    margin-top: 12px !important;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08) !important;
}

#prompt_box textarea,
#prompt_box input {
    background: #fffaf5 !important;
}

/* GPU SELECTOR */
#gpu_selector .wrap {
    display: flex !important;
    flex-direction: column !important;
    gap: 4px !important;
}

#gpu_selector label {
    font-size: 13px !important;
    padding: 4px 6px !important;
    border-radius: 6px !important;
    cursor: pointer !important;
}

#gpu_selector label:hover {
    background: #f0f4ff !important;
}

/* BENCHMARK HTML TABLE */
#benchmark_html_table {
    border: 1px solid #cbd5e1;
    border-radius: 12px;
    overflow: visible;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    max-width: 100%;
}

#benchmark_html_table table {
    border-collapse: collapse;
    width: 100%;
    font-size: 13px;
}

#benchmark_html_table thead tr {
    background: #dbeafe;
}

#benchmark_html_table thead th {
    padding: 8px 10px;
    text-align: left;
    font-weight: 600;
    white-space: nowrap;
    border: 1px solid #d6e3f5;
}

#benchmark_html_table tbody tr:nth-child(odd) {
    background: #f8fbff;
}

#benchmark_html_table tbody tr:nth-child(even) {
    background: #edf4ff;
}

#benchmark_html_table tbody td {
    padding: 6px 10px;
    border: 1px solid #d6e3f5;
    white-space: nowrap;
}

#benchmark_html_table .help-icon-th {
    margin-left: 4px;
    vertical-align: middle;
    cursor: help;
}

#benchmark_html_table td.vram-cell {
    white-space: normal;
    word-break: break-word;
    min-width: 160px;
}
""",
    )
except KeyboardInterrupt:
    cleanup_and_exit()
