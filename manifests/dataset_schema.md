# Dataset Manifest Schema

Each line of `manifest.jsonl` stores one JSON object with the following keys:

```json
{
  "sample_id": "clp4k_000001",
  "image_path": "images/clp4k_000001.png",
  "structure_map_path": "structure_maps/clp4k_000001.png",
  "style_reference_path": "style_refs/dongyuan/clp4k_000001.png",
  "width": 1536,
  "height": 1536,
  "source": "museum_public_domain",
  "painter": "unknown",
  "era": "Song-style",
  "style_label": "米氏云山式",
  "brushwork_label": "淡墨皴擦",
  "dense_caption_zh": "画面表现云雾缭绕的秋山，前景有松树与溪亭，中景山径隐现，远山层叠，整体以淡墨浅绛营造空灵意境。",
  "dense_caption_en": "A misty autumn Chinese landscape painting with pine trees and a riverside pavilion in the foreground, a mountain path in the midground, and layered distant mountains rendered in light ink and pale ochre tones.",
  "prompt_short_zh": "秋山，云雾，松树，溪亭，淡墨，浅绛，大留白",
  "prompt_structured_zh": "题材对象：秋山；构图层次：近景松树与溪亭，中景山径，远景层叠山峦；笔墨浓淡：淡墨；设色：浅绛；留白/气韵：大留白、空灵；季节天气：秋日云雾。",
  "prompt_structured_en": "Subject: autumn mountains; Composition: pine trees and a pavilion in the foreground, a mountain path in the midground, layered ranges in the background; Ink tone: light ink; Color wash: pale ochre; Blankness and mood: ample blank space, airy; Season and weather: misty autumn.",
  "structure_channels": ["lineart", "quantized_depth", "blank_space_mask", "salient_composition_mask"],
  "split": "train"
}
```

## Notes

- `structure_map_path` is a 4-channel PNG with channels in the fixed order listed in `structure_channels`.
- `style_reference_path` may be empty for samples without a curated style-reference image.
- The prompt fields are duplicated in Chinese and English so the study can compare prompt language effects.
