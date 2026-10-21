from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from PIL import Image
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoProcessor, ViltForQuestionAnswering


class ViltMultipleChoiceVQA:
    def __init__(self, model_name: str = "dandelin/vilt-b32-finetuned-vqa") -> None:
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = ViltForQuestionAnswering.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.sbert = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=str(self.device))

    def answer(self, image_path: str | Path, question: str, choices: list[str]) -> dict[str, str]:
        image = Image.open(image_path).convert("RGB")
        encoding = self.processor(images=image, text=question, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**encoding)
        predicted_class_idx = outputs.logits.argmax(-1).item()
        free_form = str(self.model.config.id2label[predicted_class_idx]).strip().lower()
        normalized_choices = [choice.strip().lower() for choice in choices]
        if free_form in normalized_choices:
            return {"free_form_answer": free_form, "multiple_choice_answer": free_form}

        embeddings = self.sbert.encode([free_form, *normalized_choices], convert_to_tensor=True)
        sims = util.cos_sim(embeddings[0], embeddings[1:])[0]
        best_idx = int(torch.argmax(sims).item())
        return {
            "free_form_answer": free_form,
            "multiple_choice_answer": normalized_choices[best_idx],
        }


def main(question_answer_path: str, id2img_path: str, output_path: str) -> None:
    question_answer_pairs = json.loads(Path(question_answer_path).read_text(encoding="utf-8"))
    id2img = json.loads(Path(id2img_path).read_text(encoding="utf-8"))
    id2img_root = Path(id2img_path).parent
    vqa = ViltMultipleChoiceVQA()

    score_by_caption: defaultdict[str, list[int]] = defaultdict(list)
    score_by_type: defaultdict[str, list[int]] = defaultdict(list)
    question_logs: defaultdict[str, dict] = defaultdict(dict)

    for qa in question_answer_pairs:
        caption_id = str(qa["id"])
        image_rel = id2img[caption_id]
        image_path = Path(image_rel)
        if not image_path.is_absolute():
            image_path = id2img_root / image_rel
        result = vqa.answer(image_path, qa["question"], qa["choices"])
        multiple_choice_answer = result["multiple_choice_answer"]
        correct = int(multiple_choice_answer == str(qa["answer"]).strip().lower())
        row = dict(qa)
        row["free_form_vqa"] = result["free_form_answer"]
        row["multiple_choice_vqa"] = multiple_choice_answer
        row["scores"] = correct
        question_logs[caption_id][qa["question"]] = row
        score_by_caption[caption_id].append(correct)
        score_by_type[qa["element_type"]].append(correct)

    averaged_scores = [mean(values) for values in score_by_caption.values()]
    summary = {
        "tifa_average": mean(averaged_scores) if averaged_scores else 0.0,
        "tifa_stdev": stdev(averaged_scores) if len(averaged_scores) > 1 else 0.0,
        "accuracy_by_type": {key: mean(values) for key, values in score_by_type.items()},
        "caption_scores": {key: mean(values) for key, values in score_by_caption.items()},
        "question_details": question_logs,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-answer-path", required=True)
    parser.add_argument("--id2img-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()
    main(
        question_answer_path=args.question_answer_path,
        id2img_path=args.id2img_path,
        output_path=args.output_path,
    )
