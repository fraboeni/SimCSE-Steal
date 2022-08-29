# inspired by: https://github.com/cleverhans-lab/SimCLR/blob/neurips/steal.py
import torch
import torch.backends.cudnn as cudnn
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
from datasets import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import tqdm
import torch.nn.functional as F


def info_nce_loss(self, features):
    n = int(features.size()[0] / self.args.batch_size)
    labels = torch.cat(
        [torch.arange(self.args.batch_size) for i in range(n)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(self.args.device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(
        similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape
    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives
    negatives = similarity_matrix[~labels.bool()].view(
        similarity_matrix.shape[0], -1)
    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(
        self.args.device)
    logits = logits / self.args.temperature
    return logits, labels


class CustomDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sample = {'sentence': row}
        return sample


def main():

    batch_size = 64
    num_queries = 10000
    train_epochs = 3
    datasetsteal = "nli"
    loss = "infonce"
    pad_to_max_length = True # whether the tokenizer should pad to max train
    max_seq_length = 32 # default value from train.py




    if torch.cuda.is_available():
        device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')
        gpu_index = -1

    if loss == "infonce":
        criterion = torch.nn.CrossEntropyLoss().to(device)
    elif loss == "mse":
        criterion = nn.MSELoss().to(device)



    # Import victim model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
    victim_model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
    victim_model.to(device)
    victim_model.eval()

    # initialize the stolen model with bert-base-uncased
    stolen_model = AutoModel.from_pretrained("bert-base-uncased")
    stolen_model.to(device)
    stolen_model.train()

    #df_sentences = pd.read_csv("../data/nli/nli_for_simcse.csv")

    # the NLI dataset has {sen1, sen2, hard_negative}
    ds = CustomDataset("../data/nli/nli_for_simcse.csv")
    train_loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    n_iter = 0

     # optimizer: #### Todo: check what optimizer SimCSE has in the code. I could not find it.
    optimizer = torch.optim.Adam(stolen_model.parameters(), # todo: check if it should be really the stolen model's parameters... But should be because we're training that.
                                 0.0001,
                                 )

    # per batch: get embeddings of a sentence for both victim and stolen model
    for epoch_counter in range(train_epochs):

        total_queries = 0

        for sent in tqdm(train_loader):
            sent_features = tokenizer(
                 sent,
                 max_length=max_seq_length,
                 truncation=True,
                 padding="max_length" if pad_to_max_length else False,
             )
            sent_features = sent_features.to(device)
            with torch.no_grad():
                stolen_embeddings = stolen_model(**sent_features, output_hidden_states=True, return_dict=True).pooler_output
            with torch.no_grad():
                victim_embeddings = victim_model(**sent_features, output_hidden_states=True,
                                                 return_dict=True).pooler_output

            stolen_embeddings = stolen_embeddings.to(device)
            victim_embeddings = victim_embeddings.to(device)

            victim_embeddings = victim_embeddings.detach() # stop gradients

            if loss == "infonce":
                all_features = torch.cat([stolen_embeddings, victim_embeddings], dim=0)
                logits, labels = info_nce_loss(all_features)
                loss = criterion(logits, labels)
            else:
                loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            print(loss)
            optimizer.step()

            n_iter += 1
            total_queries += len(sent)

            if total_queries >= num_queries:
                break

    # put model in trainer and then save
    # todo: figure out how to save the stolen model
    # print("Stealing has finished.")
    # checkpoint_name = f'stolen_checkpoint_{num_queries}_{loss}_{datasetsteal}.pth.tar'
    # save_checkpoint({
    #     'epoch': self.args.epochs,
    #     'arch': self.args.arch,
    #     'state_dict': self.model.state_dict(),
    #     'optimizer': self.optimizer.state_dict(),
    # }, is_best=False,
    #     filename=os.path.join(self.log_dir2, checkpoint_name))
    # logging.info(
    #     f"Stolen model checkpoint and metadata has been saved at {self.log_dir2}.")
    #
    #


if __name__ == '__main__':
    main()