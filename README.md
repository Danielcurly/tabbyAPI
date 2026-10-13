# TabbyAPI

<p align="left">
    <img src="https://img.shields.io/badge/Python-3.10--3.14-blue" alt="Python 3.10 through 3.14">
    <a href="/LICENSE">
        <img src="https://img.shields.io/badge/License-AGPLv3-blue.svg" alt="License: AGPL v3"/>
    </a>
    <a href="https://discord.gg/sYQxnuD7Fj">
        <img src="https://img.shields.io/discord/545740643247456267.svg?logo=discord&color=blue" alt="Discord Server"/>
    </a>
</p>

<p align="left">
    <a href="https://theroyallab.github.io/tabbyAPI">
        <img src="https://img.shields.io/badge/Documentation-API-orange" alt="Developer facing API documentation">
    </a>
</p>

<p align="left">
    <a href="https://ko-fi.com/I2I3BDTSW">
        <img src="https://img.shields.io/badge/Support_on_Ko--fi-FF5E5B?logo=ko-fi&style=for-the-badge&logoColor=white" alt="Support on Ko-Fi">
    </a>
</p>

> [!NOTE]
> **This is a fork of [theroyallab/tabbyAPI](https://github.com/theroyallab/tabbyAPI)** that exposes **equal autosplit** for ExLlamaV3 models. Everything else is upstream TabbyAPI. The feature lives on the [`equal-autosplit`](https://github.com/Danielcurly/tabbyAPI/tree/equal-autosplit) branch and **requires the ExLlamaV3 fork [Danielcurly/exllamav3](https://github.com/Danielcurly/exllamav3), branch [`equal-autosplit-v1.6`](https://github.com/Danielcurly/exllamav3/tree/equal-autosplit-v1.6)**. That fork explains how the split is computed. The rest of this README is the upstream one.

## Equal autosplit (this fork)

With several GPUs, the stock `gpu_split_auto` + `autosplit_reserve` fills one card after another. The first cards end up full and the last ones half empty. The reserve doesn't count the transient memory of long prompts either, so a model that loads fine can still run out of memory on its first long request. Finding a `gpu_split` and a `max_seq_len` that work takes trial and error, and you have to do it again for every model.

This fork adds config options that hand the layer placement to ExLlamaV3's equal autosplit. Every GPU then keeps the same free VRAM, the context can be sized automatically, and the result is checked at load time.

| Option (`model:` section) | Default | Meaning |
|---|---|---|
| `equal_split_proportions` | none | One value per visible GPU, relative to the largest. `[1, 1, 1, 1]` gives every GPU the same free VRAM; a lower value keeps a GPU partly free; `0` excludes it. Replaces `gpu_split_auto`, `autosplit_reserve` and `gpu_split`. Layer split only (not tensor parallel). |
| `equal_split_margin_mb` | `1024` | Free VRAM, in MB, that every GPU must keep. One number applies to all GPUs; a list sets one value per GPU. |
| `equal_split_overhead_mb` | `192` | Runtime overhead per GPU, in MB, added to the estimate (load buffers, kernels and workspaces after warmup). One number or one per GPU. |
| `equal_split_auto_context` | `false` | Use the largest context that fits with the margin on every GPU. A numeric `max_seq_len` becomes the upper bound. In a model's `tabby_config.yml`, `max_seq_len: auto` / `cache_size: auto` does the same. |
| `equal_split_stress_test` | `true` | After loading, run a full prompt chunk and a decode step at the end of the cache, using the draft model's real window if there is one, and check the margin on every GPU. |

How loading behaves:

- **Draft model first.** The draft model (DFlash, MTP or a separate draft) is loaded before the main model. The split and the automatic context then account for the VRAM it already takes.
- **Automatic retry.** If a GPU would end up below its margin, ExLlamaV3 raises `EqualSplitHeadroomError` with the deficit per GPU. With an automatic context, TabbyAPI shrinks the context by the suggested amount, rebuilds the caches (and reloads the draft model) and loads again, up to 3 attempts. With a fixed context, it fails with a clear error at load time instead of failing later during a request.
- **Readable logs.** The plan, the after-load comparison (estimated vs measured memory) and the stress test results are printed as per-GPU tables. On a retry the progress bars restart from the right component. A missing GPU (more proportions than visible devices) produces a clear error.

### Example

`config.yml`. This is what a 7-GPU rig runs for Qwen3.6-27B (8.0 bpw) with an MTP draft:

```yaml
model:
  equal_split_proportions: [1, 1, 1, 1, 1, 1, 1]  # one per visible GPU
  equal_split_margin_mb: 1024                     # or a list, e.g. [1536, 1024, 1024, 1024, 1024, 1024, 1024]
  equal_split_stress_test: true
  equal_split_auto_context: true                  # largest context that fits (244k tokens here)
  cache_mode: '8,8'
  chunk_size: 2048
  max_batch_size: 1

draft_model:
  draft_mode: mtp
  draft_num_tokens: 4
```

Install the ExLlamaV3 fork in TabbyAPI's environment (for example `pip install git+https://github.com/Danielcurly/exllamav3@equal-autosplit-v1.6`; it builds the extension and needs the CUDA toolkit). Then start TabbyAPI as usual.

---

> [!IMPORTANT]
>
> In addition to the README, please read the [Wiki](https://github.com/theroyallab/tabbyAPI/wiki/1.-Getting-Started) page for information about getting started!

> [!WARNING]
> The ONLY official source for TabbyAPI information is this repository. Any other websites, downloads, or services claiming to represent TabbyAPI are unaffiliated and should be considered as malicious.

> [!NOTE]
> 
> Need help? Join the [Discord Server](https://discord.gg/sYQxnuD7Fj) and get the `Tabby` role. Please be nice when asking questions.

A FastAPI based application that allows for generating text using an LLM (large language model) using the [Exllamav3](https://github.com/turboderp-org/exllamav3) backend.

TabbyAPI is also the official API backend server for ExllamaV3.

## Disclaimer

This project is marked as rolling release. There may be bugs and changes down the line. Please be aware that you might need to reinstall dependencies if needed.

TabbyAPI is a hobby project made for a small amount of users. It is not meant to run on production servers. For that, please look at other solutions that support those workloads.

## Getting Started

> [!IMPORTANT]
> 
> Looking for more information? Check out the Wiki.

For a step-by-step guide, choose the format that works best for you:

📖 Read the [Wiki](https://github.com/theroyallab/tabbyAPI/wiki/01.-Getting-Started) – Covers installation, configuration, API usage, and more.

🎥 Watch the [Video Guide](https://www.youtube.com/watch?v=03jYz0ijbUU) – A hands-on walkthrough to get you up and running quickly.

### Docker

TabbyAPI publishes a CUDA image to GitHub Container Registry. Install Docker and the NVIDIA container toolkit, then start the published image with your models directory mounted into the container:

```bash
docker pull ghcr.io/theroyallab/tabbyapi:latest
docker run --gpus all --shm-size=8g --name tabbyapi -p 5000:5000 -v /path/to/models:/app/models ghcr.io/theroyallab/tabbyapi:latest
```

Replace `/path/to/models` with the folder that contains your local model directories. The API is exposed on `http://localhost:5000`. Keep the `--shm-size` flag: ExLlamaV3 uses POSIX shared memory for tensor parallelism and CPU MoE offload, and Docker's default of 64 MiB is too small for either.

Available tags: `latest` (CUDA 12.8), `cu13` (CUDA 13), `latest-rocm` (AMD, Radeon RX 7900 series; other GPUs build from `docker/Dockerfile.rocm`), and `latest-extras` (CUDA 12.8 with the optional embeddings stack included). When building from source, pass `--build-arg EXTRAS=1` to include the embeddings stack.

For Docker Compose, custom config mounts, or building the image locally, see the [Docker instructions](docs/01.-Getting-Started.md#docker).

## Features

- OpenAI compatible API
- Loading/unloading models
- HuggingFace model downloading
- Embedding model support
- JSON schema + Regex + EBNF support
- AI Horde support
- Speculative decoding via draft models
- Inbuilt proxy to override client request parameters/samplers
- Flexible Jinja2 template engine for chat completions that conforms to HuggingFace
- Concurrent inference with asyncio
- Utilizes modern python paradigms
- Continuous batching engine using paged attention
- OAI style tool/function calling

And much more. If something is missing here, PR it in!

## Supported Model Types

TabbyAPI uses Exllama as a powerful and fast backend for model inference, loading, etc. Therefore, the following types of models are supported:

- Exl3 (Highly recommended)

- FP16/BF16

In addition, TabbyAPI supports parallel batching using paged attention for Nvidia Ampere GPUs and higher.

## Contributing

Use the template when creating issues or pull requests, otherwise the developers may not look at your post.

If you have issues with the project:

- Describe the issue in detail

- If you have a feature request, please indicate it as such.

If you have a Pull Request

- Describe the pull request in detail, what, and why you are changing something

## Acknowldgements

TabbyAPI would not exist without the work of other contributors and FOSS projects:

- [ExllamaV2](https://github.com/turboderp-org/exllamav2)
- [ExllamaV3](https://github.com/turboderp-org/exllamav3)
- [Aphrodite Engine](https://github.com/PygmalionAI/Aphrodite-engine)
- [infinity-emb](https://github.com/michaelfeil/infinity)
- [FastAPI](https://github.com/fastapi/fastapi)
- [Text Generation WebUI](https://github.com/oobabooga/text-generation-webui)
- [SillyTavern](https://github.com/SillyTavern/SillyTavern)

## Developers and Permissions

Creators/Developers:

- [kingbri](https://github.com/bdashore3)

- [Splice86](https://github.com/Splice86)

- [Turboderp](https://github.com/turboderp)
