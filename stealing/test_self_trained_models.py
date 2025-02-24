import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer


def main():
    # Import our models. The package will take care of downloading the models automatically
    tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
    #model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased") # their model
    model = AutoModel.from_pretrained("/ssd003/home/fraboeni/models/nlp-stealing/my-sup-simcse-bert-base-uncased-flickr30k-287825")

    # Tokenize input texts
    texts = [
        "There's a kid on a skateboard.",
        "A kid is skateboarding.",
        "A kid is inside the house."
    ]
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    # Get the embeddings
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

    # Calculate cosine similarities
    # Cosine similarities are in [-1, 1]. Higher means more similar
    cosine_sim_0_1 = 1 - cosine(embeddings[0], embeddings[1])
    cosine_sim_0_2 = 1 - cosine(embeddings[0], embeddings[2])

    print("Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (texts[0], texts[1], cosine_sim_0_1))
    print("Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (texts[0], texts[2], cosine_sim_0_2))


if __name__ == '__main__':
    main()