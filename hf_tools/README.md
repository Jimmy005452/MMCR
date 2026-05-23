# Hugging Face Checkpoint Tools

Tools for inspecting and converting Hugging Face CLIP checkpoints into the timm encoder format used by this project.

## Compare checkpoint keys

```bash
python -m hf_tools.compare_hf_timm_checkpoint --repo-id tanganke/clip-vit-large-patch14_sun397 --download-root checkpoint_another --report checkpoint_another/tanganke_sun397_compare_report.json
```

## Convert checkpoints

```bash
python -m hf_tools.convert_hf_clip_to_timm --download-root checkpoint_another --output-root checkpoint_another/converted --arch vit_large_patch14_clip_224.openai --overwrite
```
