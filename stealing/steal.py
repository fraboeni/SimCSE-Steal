# inspired by: https://github.com/cleverhans-lab/SimCLR/blob/neurips/steal.py
import argparse
import torch
import torch.backends.cudnn as cudnn
import os
from torchvision import models
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer



parser = argparse.ArgumentParser(description='PyTorch SimCSE')
parser.add_argument('-data', metavar='DIR',
                    default=f"/ssd003/home/{os.getenv('USER')}/data",
                    help='path to dataset')
parser.add_argument('--dataset', default='nli',
                    help='dataset name',
                    choices=['nli', 'qqp', 'flickr30k'])
parser.add_argument('--datasetsteal', default='nli',
                    help='dataset used for querying the victim',
                    choices=['nli', 'qqp', 'flickr30k'])
# base model architecture
# parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet34',
#                     choices=model_names,
#                     help='model architecture: ' +
#                          ' | '.join(model_names) +
#                          ' (default: resnet50)')
# parser.add_argument('--archstolen', default='resnet34',
#                     choices=model_names,
#                     help='stolen model architecture: ' +
#                          ' | '.join(model_names) +
#                          ' (default: resnet34)')
parser.add_argument('--epochstrain', default=200, type=int, metavar='N',
                    help='number of epochs victim was trained with')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
# todo: embedding size of our model
# parser.add_argument('--out_dim', default=128, type=int,
#                     help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=200, type=int,
                    help='Log every n steps')
parser.add_argument('--num_queries', default=9000, type=int, metavar='N',
                    help='Number of queries to steal the model.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument('--logdir', default='test', type=str,
                    help='Log directory to save output to.')
parser.add_argument('--lossvictim', default='infonce', type=str,
                    help='Loss function victim was trained with')
parser.add_argument('--clear', default='False', type=str,
                    help='Clear previous logs', choices=['True', 'False'])

def main():
    args = parser.parse_args()
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    # Import our models. The package will take care of downloading the models automatically
    tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
    #model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
    model = AutoModel.from_pretrained("/h/321/fraboeni/code/SimCSE-Steal/result/my-sup-simcse-bert-base-uncased-qqp")

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