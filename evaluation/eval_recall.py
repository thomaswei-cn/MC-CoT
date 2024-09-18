from transformers import AutoTokenizer



def calculate(ref, pred, tokenizer):
    ref_encoded = set(tokenizer.encode(str(ref)))
    pred_encoded = set(tokenizer.encode(str(pred)))
    recall = len(pred_encoded.intersection(ref_encoded)) / len(ref_encoded)
    return recall


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    ref = "Chest X-ray"
    pred = """Rationale: The description provided indicates that the image is an X-ray of a person's chest, specifically showing the heart and surrounding structures. The characteristics mentioned, such as the black and white appearance with varying shades of gray, the positioning of the heart in the chest cavity, and the presence of potential abnormalities like an enlarged heart and a wire indicating a medical device, all align with typical features seen in X-ray images of the chest.

Answer: This is an X-ray image of the chest."""
    recall= calculate(ref, pred, tokenizer)
    print(f"Recall: {recall}")
