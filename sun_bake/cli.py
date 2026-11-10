import os
import sys
import argparse
import json
import cv2

from .core.model import MocModel
from .core.reconstructor import export_model_to_psd, ExportOptions
from .core.baker import bake_psd_to_atlases, BakeOptions
from .core.preview import render_model_preview


def load_model(model_arg, texture_args=None, dll_path=None, expression_arg=None) -> MocModel:
    if not os.path.isfile(model_arg):
        raise FileNotFoundError(f"Model not found: {model_arg}")

    if model_arg.lower().endswith(".json"):
        return MocModel.from_model3_json(model_arg, dll_path=dll_path, expression=expression_arg)
    return MocModel(moc_path=model_arg, texture_paths=texture_args, dll_path=dll_path, expression=expression_arg)


def _progress(cur, tot, msg):
    pct = (cur / tot) * 100 if tot > 0 else 0
    sys.stdout.write(f"\r[{pct:5.1f}%] {msg:<50}")
    sys.stdout.flush()


def handle_export(args):
    model = load_model(args.model, args.textures, args.dll_path, args.expression)
    print(f"Loaded {os.path.basename(model.moc_path)} ({model.canvas.width:.0f}x{model.canvas.height:.0f}, {len(model.drawables)} drawables)")
    if model.current_expression:
        print(f"Expression: {model.current_expression}")

    out_psd = args.output
    if not out_psd:
        base = os.path.splitext(os.path.basename(model.moc_path))[0]
        out_psd = os.path.join(os.path.dirname(model.moc_path), f"{base}.psd")

    opts = ExportOptions(
        group_by_part=not args.flat,
        write_metadata=not args.no_meta,
        expression=args.expression,
    )

    res = export_model_to_psd(model, out_psd, opts, progress_callback=_progress)
    print(f"\nSaved {out_psd} ({res['exported_layers']} layers, {res['file_size_mb']:.1f} MB, {res['elapsed_seconds']:.2f}s)")


def handle_bake(args):
    psd_path = os.path.abspath(args.psd)
    if not os.path.isfile(psd_path):
        raise FileNotFoundError(f"PSD not found: {psd_path}")

    model_path = args.model
    expr = args.expression

    meta_path = f"{psd_path}.meta.json"
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if not model_path and meta.get("model_moc"):
            model_path = os.path.normpath(os.path.join(os.path.dirname(psd_path), meta["model_moc"]))
        if not expr and meta.get("expression"):
            expr = meta["expression"]

    if not model_path:
        print("Error: specify -m/--model or place a .psd.meta.json next to the PSD.", file=sys.stderr)
        sys.exit(1)

    model = load_model(model_path, args.textures, args.dll_path, expr)
    opts = BakeOptions(
        bleed_radius=args.bleed,
        backup_original=not args.no_backup,
        output_dir=args.output_dir,
        expression=expr,
    )

    res = bake_psd_to_atlases(psd_path, model, opts, progress_callback=_progress)
    print(f"\nBaked {res['baked_layers']} layers ({res['unchanged_layers']} unchanged) in {res['elapsed_seconds']:.2f}s")
    for p in res["written_textures"]:
        print(f"  {p}")


def handle_preview(args):
    model = load_model(args.model, args.textures, args.dll_path, args.expression)
    out_img = args.output or "model_preview.png"
    preview = render_model_preview(model, scale=args.scale)
    cv2.imwrite(out_img, preview)
    print(f"Saved {out_img} ({preview.shape[1]}x{preview.shape[0]})")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sun-bake", description="Moc3 <-> PSD converter and texture atlas baker")
    sub = parser.add_subparsers(dest="command")

    p_exp = sub.add_parser("export", help="Export model to PSD")
    p_exp.add_argument("model", help=".model3.json or .moc3 path")
    p_exp.add_argument("-o", "--output", help="Output .psd path")
    p_exp.add_argument("-e", "--expression", help="Expression name or .exp3.json path")
    p_exp.add_argument("-t", "--textures", nargs="+", help="Texture paths")
    p_exp.add_argument("--flat", action="store_true", help="Flat layer order (no part folders)")
    p_exp.add_argument("--no-meta", action="store_true", help="Skip .psd.meta.json")
    p_exp.add_argument("--dll-path", help="Live2DCubismCore.dll path")

    p_bake = sub.add_parser("bake", help="Bake edited PSD back to textures")
    p_bake.add_argument("psd", help="Edited .psd path")
    p_bake.add_argument("-m", "--model", help=".model3.json or .moc3 path")
    p_bake.add_argument("-e", "--expression", help="Expression name or .exp3.json path")
    p_bake.add_argument("-o", "--output-dir", help="Output directory")
    p_bake.add_argument("-t", "--textures", nargs="+", help="Texture paths")
    p_bake.add_argument("-b", "--bleed", type=int, default=2, help="Edge bleed px (default: 2)")
    p_bake.add_argument("--no-backup", action="store_true", help="Skip .bak files")
    p_bake.add_argument("--dll-path", help="Live2DCubismCore.dll path")

    p_prev = sub.add_parser("preview", help="Render model preview")
    p_prev.add_argument("model", help=".model3.json or .moc3 path")
    p_prev.add_argument("-o", "--output", help="Output PNG path")
    p_prev.add_argument("-e", "--expression", help="Expression name or .exp3.json path")
    p_prev.add_argument("-s", "--scale", type=float, default=0.25, help="Scale factor (default: 0.25)")
    p_prev.add_argument("-t", "--textures", nargs="+", help="Texture paths")
    p_prev.add_argument("--dll-path", help="Live2DCubismCore.dll path")

    sub.add_parser("gui", help="Open GUI")

    args = parser.parse_args(argv)
    if args.command == "export":
        handle_export(args)
    elif args.command == "bake":
        handle_bake(args)
    elif args.command == "preview":
        handle_preview(args)
    else:
        from .gui.main_window import run_gui
        run_gui()


if __name__ == "__main__":
    main()
