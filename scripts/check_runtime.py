"""Check locked imports and a tiny offline model workload; CUDA is required by default."""

import argparse
import json
import platform
import tomllib
from importlib import metadata
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true", help="run preliminary checks without a GPU")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]["name"]
    package = project.replace("-", "_")
    __import__(package + ".config")
    __import__(package + ".tracking")
    versions = {
        name: metadata.version(name)
        for name in ["torch", "transformers", "peft", "accelerate", "datasets"]
    }
    locked = tomllib.loads((root / "uv.lock").read_text())["package"]
    for name, version in versions.items():
        wanted = next(p["version"] for p in locked if p["name"] == name)
        if version.split("+")[0] != wanted.split("+")[0]:
            raise RuntimeError(f"{name}: installed {version}, lockfile requires {wanted}")
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --cpu only for preliminary local checks")
    device = "cpu" if args.cpu else "cuda"
    torch.manual_seed(7)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    # A tiny random model of the family each project trains: Qwen2.5 here (Qwen2 architecture).
    config_class, model_class = (
        (Qwen3Config, Qwen3ForCausalLM)
        if project == "residual-disposition"
        else (Qwen2Config, Qwen2ForCausalLM)
    )
    config = config_class(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    config._attn_implementation = "eager"
    model = model_class(config).to(device)
    if project == "residual-disposition":
        model = get_peft_model(
            model,
            LoraConfig(
                r=4,
                lora_alpha=8,
                target_modules=["q_proj", "v_proj"],
                task_type="CAUSAL_LM",
            ),
        )
    else:
        model.get_input_embeddings().requires_grad_(False)
        model.get_output_embeddings().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tokens = torch.randint(0, config.vocab_size, (2, 16), device=device)
    loss = model(input_ids=tokens, labels=tokens).loss
    assert torch.isfinite(loss), "Non-finite forward loss"
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(torch.count_nonzero(g).item() for g in gradients), "All gradients are zero"
    optimizer.step()
    report = dict(
        project=project,
        device=device,
        python=platform.python_version(),
        packages=versions,
        cuda_runtime=torch.version.cuda,
        loss=loss.item(),
    )
    if device == "cuda":
        torch.cuda.synchronize()
        report["gpu"] = torch.cuda.get_device_name(0)
        report["compute_capability"] = torch.cuda.get_device_capability(0)
    if project == "residual-disposition":
        import trl
        import vllm

        report["packages"].update(trl=trl.__version__, vllm=vllm.__version__)
        if device == "cuda":
            from vllm import _custom_ops

            value = torch.randn(8, 64, device=device, dtype=torch.bfloat16)
            weight = torch.ones(64, device=device, dtype=torch.bfloat16)
            output = torch.empty_like(value)
            # Check the compiled extension directly, without creating an inference engine.
            _custom_ops.rms_norm(output, value, weight, 1e-6)
            assert output.shape == value.shape and torch.isfinite(output).all()
            reference = value.float() * torch.rsqrt(
                value.float().square().mean(-1, keepdim=True) + 1e-6
            )
            torch.testing.assert_close(output.float(), reference, rtol=0.02, atol=0.02)
            report["vllm_cuda_kernel"] = "passed"
    report["status"] = "passed"
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
