import os
import json
import glob
import sentencepiece as spm

DATA_DIR = "data"
OUTPUT_TEXT = "dataset.txt"

VOCAB_SIZE = 8000
MODEL_PREFIX = "opensoftware_world_osw1_tokenizer"

def load_json_pairs(json_dir):
    texts = []
    if not os.path.isdir(json_dir):
        return texts

    for path in glob.glob(os.path.join(json_dir, "*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"⚠️ {path} Unreadable: {e}")
            continue

        intents = data.get("intents", data if isinstance(data, list) else [])

        for intent in intents:
            for p in intent.get("patterns", []):
                texts.append(p)

            for r in intent.get("responses", []):
                texts.append(r)

    return texts

def load_txt_qa_pairs(qa_dir):
    texts = []
    if not os.path.isdir(qa_dir):
        return texts

    for path in glob.glob(os.path.join(qa_dir, "*.txt")):
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()

            if line.startswith("Q:"):
                texts.append(line[2:].strip())

            elif line.startswith("A:"):
                texts.append(line[2:].strip())

    return texts

def load_plain_texts(txt_dir):
    texts = []
    if not os.path.isdir(txt_dir):
        return texts

    for path in glob.glob(os.path.join(txt_dir, "*.txt")):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()

            if content:
                texts.append(content)

    return texts

print("📚 Reading training data...")

all_texts = []

all_texts.extend(load_json_pairs(os.path.join(DATA_DIR, "json")))
all_texts.extend(load_txt_qa_pairs(os.path.join(DATA_DIR, "txt_qa")))
all_texts.extend(load_plain_texts(os.path.join(DATA_DIR, "txt")))

if len(all_texts) == 0:
    raise RuntimeError("No training data was found.")

print(f"✅ Total number of texts: {len(all_texts)}")

with open(OUTPUT_TEXT, "w", encoding="utf-8") as f:
    for text in all_texts:
        f.write(text.replace("\n", " ") + "\n")

print(f"📝 {OUTPUT_TEXT} was created.")

print("🧠 The SentencePiece tokenizer is being trained...")

spm.SentencePieceTrainer.train(
    input=OUTPUT_TEXT,
    model_prefix=MODEL_PREFIX,
    vocab_size=VOCAB_SIZE,
    hard_vocab_limit=False,
    model_type="unigram",

    character_coverage=1.0,

    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,

    shuffle_input_sentence=True,

    pad_piece="<pad>",
    unk_piece="<unk>",
    bos_piece="<bos>",
    eos_piece="<eos>",

    train_extremely_large_corpus=True
)

print("\n🎉 Completed!")
print(f"Model : {MODEL_PREFIX}.model")
print(f"Vocab : {MODEL_PREFIX}.vocab")