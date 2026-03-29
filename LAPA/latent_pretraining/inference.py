import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont



class FLAGSClass:
    def __init__(self, flag_dict):
        for key, value in flag_dict.items():
            setattr(self, key, value)


def _maybe_restart_with_cpu(exc: Exception, context: str) -> bool:
    try:
        from latent_pretraining.runtime_compat import maybe_restart_with_cpu

        return maybe_restart_with_cpu(exc, context)
    except Exception:
        return False

class LAPAInference:
    def __init__(
        self,
        image_size: int = 256,
        enable_model: bool = True,
        **kwargs,
    ) -> None:
        flags = FLAGSClass(kwargs)
        self.model = None
        if enable_model:
            from latent_pretraining.sampler_latent_pretrain import DeltaSampler

            self.model = DeltaSampler(FLAGS=flags)
        self.image_size = image_size
        self.tokens_per_delta = kwargs['tokens_per_delta']
        self.task_description = None

    def inference(self, image: np.ndarray, task_description: Optional[str] = None, *args, **kwargs) -> np.ndarray:
        assert image.dtype == np.uint8
        if self.model is None:
            return np.asarray(_mock_latent_tokens(image, task_description or "", self.tokens_per_delta), dtype=np.int32)

        image = Image.fromarray(image)
        prompts = [{'image': [image], 'question': task_description}]

        latent_output = self.model(prompts)
        latent_action = latent_output[0]

        return latent_action


def _flatten_latent_action(value: Any) -> list[int]:
    if isinstance(value, np.ndarray):
        flat = value.reshape(-1).tolist()
        return [int(x) for x in flat]
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_flatten_latent_action(item))
        return out
    try:
        return [int(value)]
    except Exception:
        return []


def _mock_latent_tokens(image: np.ndarray, instruction: str, token_count: int) -> list[int]:
    safe_count = max(1, int(token_count))
    image_mean = int(np.mean(image))
    text_score = sum(ord(ch) for ch in instruction) % 100_000
    seed = image_mean * 31 + text_score
    rng = np.random.default_rng(seed)
    return rng.integers(0, 8, size=safe_count, dtype=np.int32).tolist()


def _draw_wrapped_text(draw: ImageDraw.ImageDraw, text: str, x: int, y: int, max_chars: int, fill, font) -> int:
    words = text.split()
    line = ""
    line_count = 0
    for word in words:
        trial = (line + " " + word).strip()
        if len(trial) <= max_chars:
            line = trial
            continue
        draw.text((x, y + line_count * 16), line, fill=fill, font=font)
        line_count += 1
        line = word
    if line:
        draw.text((x, y + line_count * 16), line, fill=fill, font=font)
        line_count += 1
    return line_count


def _build_visual_frame(image: np.ndarray, instruction: str, tokens: list[int], shown_token_count: int) -> np.ndarray:
    base = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    width, height = base.size
    footer_h = 120
    canvas = Image.new("RGB", (width, height + footer_h), (22, 24, 28))
    canvas.paste(base, (0, 0))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text((12, height + 8), "Instruction:", fill=(180, 220, 255), font=font)
    used = _draw_wrapped_text(draw, instruction, 100, height + 8, max_chars=72, fill=(240, 240, 240), font=font)
    shown = tokens[: max(1, min(shown_token_count, len(tokens)))]
    token_text = " ".join(str(tok) for tok in shown)
    draw.text((12, height + 12 + used * 16), f"Latent tokens ({len(shown)}/{len(tokens)}):", fill=(180, 220, 255), font=font)
    _draw_wrapped_text(draw, token_text, 12, height + 28 + used * 16, max_chars=95, fill=(255, 255, 180), font=font)
    return np.asarray(canvas, dtype=np.uint8)


def _save_visualization_image(image: np.ndarray, instruction: str, tokens: list[int], out_path: Path) -> None:
    frame = _build_visual_frame(image, instruction, tokens, shown_token_count=len(tokens))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(out_path)


def _save_visualization_video(
    image: np.ndarray,
    instruction: str,
    tokens: list[int],
    out_path: Path,
    fps: int,
    min_frames: int,
) -> None:
    try:
        import imageio.v2 as imageio
    except Exception as exc:
        raise RuntimeError("imageio[ffmpeg] is required for mp4 output.") from exc

    total_tokens = max(1, len(tokens))
    frame_count = max(min_frames, total_tokens)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        str(out_path),
        fps=max(1, int(fps)),
        codec="libx264",
        quality=8,
        macro_block_size=1,
    ) as writer:
        for idx in range(1, frame_count + 1):
            shown = int(np.ceil(idx * total_tokens / frame_count))
            frame = _build_visual_frame(image, instruction, tokens, shown_token_count=shown)
            writer.append_data(frame)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LAPA Inference script")
    parser.add_argument('--tokens_per_delta', type=int, default=4, help='Tokens per delta')
    parser.add_argument('--vqgan_checkpoint', type=str, default="lapa_checkpoints/vqgan")
    parser.add_argument('--vocab_file', type=str, default='lapa_checkpoints/tokenizer.model')
    parser.add_argument('--multi_image', type=int, default=1)
    parser.add_argument('--jax_distributed', type=dict, default=None)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--mesh_dim', type=str, default="1,-1,1,1")
    parser.add_argument('--dtype', type=str, default="bf16")
    parser.add_argument('--load_llama_config', type=str, default="7b")
    parser.add_argument('--update_llama_config', type=str, default="dict(delta_vocab_size=8,sample_mode='text',theta=50000000,max_sequence_length=2048,scan_attention=True,scan_query_chunk_size=128,scan_key_chunk_size=128,scan_mlp=True,scan_mlp_chunk_size=1024,scan_layers=True)")
    parser.add_argument('--load_checkpoint', type=str, default="params::lapa_checkpoints/params")
    parser.add_argument('--codebook_size', type=int, default=8)

    parser.add_argument('--input-image', type=str, default='imgs/bridge_inference.jpg', help='Input image path')
    parser.add_argument('--instruction', type=str, default='move the object', help='Task instruction')
    parser.add_argument('--output-dir', type=str, default='outputs/inference', help='Directory for json/png/mp4 outputs')
    parser.add_argument('--save-video', dest='save_video', action='store_true')
    parser.add_argument('--no-save-video', dest='save_video', action='store_false')
    parser.set_defaults(save_video=True)
    parser.add_argument('--save-visualization', dest='save_visualization', action='store_true')
    parser.add_argument('--no-save-visualization', dest='save_visualization', action='store_false')
    parser.set_defaults(save_visualization=True)
    parser.add_argument('--video-fps', type=int, default=6)
    parser.add_argument('--video-min-frames', type=int, default=16)
    parser.add_argument('--mock', action='store_true', help='Skip model loading and emit deterministic mock latent tokens')
    parser.add_argument('--allow-mock-on-error', action='store_true', help='Fallback to mock tokens if model inference fails')
    return parser.parse_args()


def _load_model_configs(args: argparse.Namespace) -> argparse.Namespace:
    from latent_pretraining.delta_llama import VideoLLaMAConfig

    args.tokenizer = VideoLLaMAConfig.get_tokenizer_config()
    args.llama = VideoLLaMAConfig.get_default_config()
    args.tokenizer.vocab_file = args.vocab_file
    return args


def _main() -> None:
    args = _parse_args()
    image = np.array(Image.open(args.input_image).convert("RGB"), dtype=np.uint8)
    instruction = args.instruction

    latent_action = None
    mode = "model"
    model_error = None

    if args.mock:
        mode = "mock"
        latent_action = np.asarray(_mock_latent_tokens(image, instruction, args.tokens_per_delta), dtype=np.int32)
    else:
        try:
            args = _load_model_configs(args)
            from tux import JaxDistributedConfig, set_random_seed

            if args.jax_distributed is None:
                args.jax_distributed = JaxDistributedConfig.get_default_config()
            JaxDistributedConfig.initialize(args.jax_distributed)
            set_random_seed(args.seed)

            lapa = LAPAInference(
                image_size=256,
                enable_model=True,
                tokens_per_delta=args.tokens_per_delta,
                vqgan_checkpoint=args.vqgan_checkpoint,
                vocab_file=args.vocab_file,
                multi_image=args.multi_image,
                jax_distributed=args.jax_distributed,
                seed=args.seed,
                mesh_dim=args.mesh_dim,
                dtype=args.dtype,
                load_llama_config=args.load_llama_config,
                update_llama_config=args.update_llama_config,
                load_checkpoint=args.load_checkpoint,
                tokenizer=args.tokenizer,
                llama=args.llama,
            )
            latent_action = lapa.inference(image, instruction)
        except Exception as exc:
            if not args.allow_mock_on_error:
                raise
            model_error = f"{type(exc).__name__}: {exc}"
            mode = "mock-fallback"
            latent_action = np.asarray(_mock_latent_tokens(image, instruction, args.tokens_per_delta), dtype=np.int32)
            print("[warn] Model inference failed. Using deterministic mock latent tokens.")
            print(f"[warn] {model_error}")

    tokens = _flatten_latent_action(latent_action)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "latent_action.json"
    png_path = output_dir / "latent_action_visualization.png"
    mp4_path = output_dir / "latent_action_visualization.mp4"

    payload = {
        "mode": mode,
        "instruction": instruction,
        "input_image": str(args.input_image),
        "tokens": tokens,
        "token_count": len(tokens),
    }
    if model_error is not None:
        payload["model_error"] = model_error

    with json_path.open("w", encoding="utf-8") as fout:
        json.dump(payload, fout, indent=2)

    if args.save_visualization:
        _save_visualization_image(image, instruction, tokens, png_path)

    if args.save_video:
        _save_visualization_video(
            image,
            instruction,
            tokens,
            out_path=mp4_path,
            fps=args.video_fps,
            min_frames=args.video_min_frames,
        )

    print("latent action tokens:", tokens)
    print("saved json:", json_path)
    if args.save_visualization:
        print("saved visualization:", png_path)
    if args.save_video:
        print("saved video:", mp4_path)


if __name__ == "__main__":
    try:
        _main()
    except Exception as exc:
        if not _maybe_restart_with_cpu(exc, "latent_pretraining.inference"):
            raise
