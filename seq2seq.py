import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import spacy
import datasets
import torchtext
import tqdm
import evaluate
import pickle as pkl
import selfies as sf

seed = 1234

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

#######################################
#ATTEMPT AT BUILDING MY OWN TOKENISERS#
#######################################

#####################################################################################################################

#reading the dataset
dataset_path = "data/spectra.pkl"
print("hello1")
#opens the pickle file and reads it
with open(dataset_path, "rb") as f:
    dataset = pkl.load(f)

#prints molecule 1000 to test the data was correctly loaded
print(dataset["1000"])

class SpectrumTokeniser:
    def __init__(self, vocab, sos_token="<sos>", eos_token="<eos>", unk_token="<unk>"):
        self.vocab = vocab
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.inv_vocab = {idx: token for token, idx in vocab.items()}  # Reverse mapping

    def tokenise(self, spectrum_array):
        """Extract peak indices from spectrum array."""
        peak_indices = [str(idx) for idx, value in enumerate(spectrum_array) if value == 1]
        return peak_indices

    def tokenise_with_special_tokens(self, spectrum_array):
        """Tokenize and add special tokens."""
        tokens = [self.sos_token] + self.tokenise(spectrum_array) + [self.eos_token]
        return tokens

    def encode(self, spectrum_array):
        """Convert a spectrum into a tensor of token indices."""
        tokens = self.tokenise_with_special_tokens(spectrum_array)
        return torch.tensor([self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens], dtype=torch.long)

    def decode(self, token_indices):
        """Convert token indices back to spectrum representation."""
        tokens = [self.inv_vocab[idx] for idx in token_indices if idx in self.inv_vocab]
        return [int(token) for token in tokens if token not in [self.sos_token, self.eos_token]]
    
    def encode_from_tokens(self, tokens):
        return torch.tensor(
        [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens],
        dtype=torch.long
    )

# Example vocab generation
def build_spectrum_vocab(dataset):
    unique_tokens = set()
    
    for value in dataset.values():
        spectrum_array = value["binned_proton"]
        peak_indices = [str(idx) for idx, val in enumerate(spectrum_array) if val == 1]
        unique_tokens.update(peak_indices)

    special_tokens = ["<sos>", "<eos>", "<unk>", "<pad>"]
    all_tokens = special_tokens + sorted(unique_tokens)

    return {token: idx for idx, token in enumerate(all_tokens)}

# Generate vocab from dataset
spectrum_vocab = build_spectrum_vocab(dataset)

# Initialize tokeniser with vocab
spectrum_tokeniser = SpectrumTokeniser(spectrum_vocab)

# Tokenize and encode an example spectrum
example_spectrum = dataset["1000"]["binned_proton"]
encoded_spectrum = spectrum_tokeniser.encode(example_spectrum)

print("Encoded:", encoded_spectrum)  # Tensor output

# Decode back to original peak indices
decoded_spectrum = spectrum_tokeniser.decode(encoded_spectrum.tolist())
print("Decoded:", decoded_spectrum)  # List of peak indices





##############################
# TOKENISING A SELFIE STRING #
##############################

#
#SELFIE VOCAB BUILDER
# 

def selfie_vocab_builder(selfies_list):
    import selfies as sf
    from collections import defaultdict

    #Extract unique tokens
    unique_tokens = set()
    for selfies in selfies_list:
        tokens = list(sf.split_selfies(selfies))  # Tokenize the SELFIES string
        unique_tokens.update(tokens)

    #Add special tokens
    special_tokens = ["<sos>", "<eos>", "<unk>", "<pad>"]
    all_tokens = special_tokens + sorted(unique_tokens)  # Sorting ensures consistent indexing

    #Create vocab dictionary
    vocab = {token: idx for idx, token in enumerate(all_tokens)}

    return vocab


#constructing a list of all selfies
selfies_list = []

#loops through the dataset and gets the values
for value in dataset.values():
    selfies_list.append(value["selfie"])

#builds the vocab
selfie_vocab = selfie_vocab_builder(selfies_list)


#
#SEFIE TOKENISER
#

import selfies as sf

class SelfiesTokeniser:
    def __init__(self, vocab, sos_token="<sos>", eos_token="<eos>", unk_token="<unk>", pad_token = "<pad>"):
        self.vocab = vocab
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.inv_vocab = {idx: token for token, idx in vocab.items()}  # Reverse lookup for decoding

    def tokenise(self, selfies_string):
        tokens = list(sf.split_selfies(selfies_string))
        return tokens

    def tokenise_with_special_tokens(self, selfies_string):
        tokens  = [self.sos_token] + self.tokenise(selfies_string) + [self.eos_token]
        return tokens

    def encode(self, selfies_string):
        """Encodes a SELFIES string into numerical token indices."""
        tokens = self.tokenise_with_special_tokens(selfies_string)
        return torch.tensor([self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens], dtype=torch.long)

    def decode(self, token_indices):
        """Decodes a sequence of token indices back into a SELFIES string."""
        tokens = [self.inv_vocab[idx] for idx in token_indices if idx in self.inv_vocab]
        return "".join(tokens).replace(self.sos_token, "").replace(self.eos_token, "")
    
    def encode_from_tokens(self, tokens):
        return torch.tensor(
            [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens],
            dtype=torch.long
        )

# Example usage:
selfie_tokeniser = SelfiesTokeniser(selfie_vocab)

# Example SELFIES string
selfies_string = "[C][O][O][H]"

# Tokenizing
tokens = selfie_tokeniser.tokenise_with_special_tokens(selfies_string)
print("Tokens:", tokens)

# Encoding to indices
encoded = selfie_tokeniser.encode(selfies_string)
print("Encoded:", encoded)

# Decoding back to SELFIES
decoded = selfie_tokeniser.decode(encoded)
print("Decoded:", decoded)

#tokenises a selfie string as an example
example_selfie = dataset["1000"]["selfie"]  # Get the binned spectrum
tokenised_selfie = selfie_tokeniser.tokenise_with_special_tokens(example_selfie)

print(tokenised_selfie)
######################################################################################################################

#do split test and train
dataset_lenth = len(dataset)
print(f"the dataset length was {dataset_lenth}")
np.random.seed(42) 
random_array = np.random.rand(dataset_lenth)

print(random_array)


train_data = {"spectra":[], "selfies":[]}
test_data = {"spectra":[], "selfies":[]}
valid_data = {"spectra":[], "selfies":[]}

count = 0 
for molecule in dataset.values():
    if random_array[count] < 0.03125:
        #test_data.append({"spectrum":molecule["binned_carbon"], "selfie": molecule["selfie"]})
        test_data["spectra"].append(molecule["binned_carbon"])
        test_data["selfies"].append(molecule["selfie"])
    elif 0.03125 <= random_array[count] < 0.0625:
        #valid_data.append({"spectrum":molecule["binned_carbon"], "selfie": molecule["selfie"]})
        valid_data["spectra"].append(molecule["binned_carbon"])
        valid_data["selfies"].append(molecule["selfie"])
    else:
        #train_data.append({"spectrum":molecule["binned_carbon"], "selfie": molecule["selfie"]})
        train_data["spectra"].append(molecule["binned_carbon"])
        train_data["selfies"].append(molecule["selfie"])
    count += 1


# Combine spectra and selfies into list of dicts
train_data = [
    {"spectrum": spec, "selfie": self}
    for spec, self in zip(train_data["spectra"], train_data["selfies"])
]
valid_data = [
    {"spectrum": spec, "selfie": self}
    for spec, self in zip(valid_data["spectra"], valid_data["selfies"])
]
test_data = [
    {"spectrum": spec, "selfie": self}
    for spec, self in zip(test_data["spectra"], test_data["selfies"])
]


def tokenise_example(example, spectrum_tokeniser, selfie_tokeniser):
    spectrum_tokens = spectrum_tokeniser.tokenise_with_special_tokens(example["spectrum"])
    selfie_tokens = selfie_tokeniser.tokenise_with_special_tokens(example["selfie"])
    return {
        "spectrum_tokens": spectrum_tokens,
        "selfie_tokens": selfie_tokens
    }




max_length = 1_000
lower = True
sos_token = "<sos>"
eos_token = "<eos>"

fn_kwargs = {
    "spectrum_tokeniser": spectrum_tokeniser,
    "selfie_tokeniser": selfie_tokeniser
}
train_data = [tokenise_example(ex, **fn_kwargs) for ex in train_data]
valid_data = [tokenise_example(ex, **fn_kwargs) for ex in valid_data]
test_data  = [tokenise_example(ex, **fn_kwargs) for ex in test_data]


min_freq = 2
unk_token = "<unk>"
pad_token = "<pad>"

special_tokens = [
    unk_token,
    pad_token,
    sos_token,
    eos_token,
]


assert spectrum_vocab[unk_token] == selfie_vocab[unk_token]
assert spectrum_vocab[pad_token] == selfie_vocab[pad_token]

unk_index = selfie_vocab[unk_token]
pad_index = selfie_vocab[pad_token]

from torchtext.vocab import Vocab
from collections import Counter

from torchtext.vocab import build_vocab_from_iterator

def convert_to_torchtext_vocab(vocab_dict, specials=None):
    if specials is None:
        specials = ["<unk>", "<pad>", "<sos>", "<eos>"]

    # Create vocab from a list of tokens (wrapped in a list to make it iterable)
    vocab = build_vocab_from_iterator([vocab_dict.keys()], specials=specials, special_first=True)

    # Set <unk> index
    vocab.set_default_index(vocab["<unk>"])

    return vocab
special_tokens = ["<unk>", "<pad>", "<sos>", "<eos>"]

selfie_vocab = convert_to_torchtext_vocab(selfie_vocab)
spectrum_vocab = convert_to_torchtext_vocab(spectrum_vocab)

selfie_vocab.set_default_index(unk_index)
spectrum_vocab.set_default_index(unk_index)

def numericalize_example(example, spectrum_tokeniser, selfie_tokeniser):
    spectrum_ids = spectrum_tokeniser.encode_from_tokens(example["spectrum_tokens"])
    selfie_ids = selfie_tokeniser.encode_from_tokens(example["selfie_tokens"])
    return {"spectrum_ids": spectrum_ids, "selfie_ids": selfie_ids}

fn_kwargs = {
    "spectrum_tokeniser": spectrum_tokeniser,
    "selfie_tokeniser": selfie_tokeniser
}

train_data = [numericalize_example(ex, **fn_kwargs) for ex in train_data]
valid_data = [numericalize_example(ex, **fn_kwargs) for ex in valid_data]
test_data  = [numericalize_example(ex, **fn_kwargs) for ex in test_data]


from datasets import Dataset

train_data = Dataset.from_list(train_data)
valid_data = Dataset.from_list(valid_data)
test_data  = Dataset.from_list(test_data)


data_type = "torch"
format_columns = ["spectrum_ids", "selfie_ids"]

train_data = train_data.with_format(
    type=data_type, columns=format_columns, output_all_columns=True
)

valid_data = valid_data.with_format(
    type=data_type,
    columns=format_columns,
    output_all_columns=True,
)

test_data = test_data.with_format(
    type=data_type,
    columns=format_columns,
    output_all_columns=True,
)




def get_collate_fn(pad_index):
    def collate_fn(batch):
        batch_spectrum_ids = [example["spectrum_ids"] for example in batch]
        batch_selfie_ids = [example["selfie_ids"] for example in batch]
        batch_spectrum_ids = nn.utils.rnn.pad_sequence(batch_spectrum_ids, padding_value=pad_index)
        batch_selfie_ids = nn.utils.rnn.pad_sequence(batch_selfie_ids, padding_value=pad_index)
        batch = {
            "spectrum_ids": batch_spectrum_ids,
            "selfie_ids": batch_selfie_ids,
        }
        return batch

    return collate_fn


def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
    collate_fn = get_collate_fn(pad_index)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )
    return data_loader

batch_size = 128

train_data_loader = get_data_loader(train_data, batch_size, pad_index, shuffle=True)
valid_data_loader = get_data_loader(valid_data, batch_size, pad_index)
test_data_loader = get_data_loader(test_data, batch_size, pad_index)

print("PRE ENCODER PHASE COMPLETE")

# #  .----------------.  .-----------------. .----------------.  .----------------.  .----------------.  .----------------.  .----------------. 
# # | .--------------. || .--------------. || .--------------. || .--------------. || .--------------. || .--------------. || .--------------. |
# # | |  _________   | || | ____  _____  | || |     ______   | || |     ____     | || |  ________    | || |  _________   | || |  _______     | |
# # | | |_   ___  |  | || ||_   \|_   _| | || |   .' ___  |  | || |   .'    `.   | || | |_   ___ `.  | || | |_   ___  |  | || | |_   __ \    | |
# # | |   | |_  \_|  | || |  |   \ | |   | || |  / .'   \_|  | || |  /  .--.  \  | || |   | |   `. \ | || |   | |_  \_|  | || |   | |__) |   | |
# # | |   |  _|  _   | || |  | |\ \| |   | || |  | |         | || |  | |    | |  | || |   | |    | | | || |   |  _|  _   | || |   |  __ /    | |
# # | |  _| |___/ |  | || | _| |_\   |_  | || |  \ `.___.'\  | || |  \  `--'  /  | || |  _| |___.' / | || |  _| |___/ |  | || |  _| |  \ \_  | |
# # | | |_________|  | || ||_____|\____| | || |   `._____.'  | || |   `.____.'   | || | |________.'  | || | |_________|  | || | |____| |___| | |
# # | |              | || |              | || |              | || |              | || |              | || |              | || |              | |
# # | '--------------' || '--------------' || '--------------' || '--------------' || '--------------' || '--------------' || '--------------' |
# #  '----------------'  '----------------'  '----------------'  '----------------'  '----------------'  '----------------'  '----------------' 


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

        
# #      _______. _______   ______      ___        _______. _______   ______      
# #     /       ||   ____| /  __  \    |__ \      /       ||   ____| /  __  \     
# #    |   (----`|  |__   |  |  |  |      ) |    |   (----`|  |__   |  |  |  |    
# #     \   \    |   __|  |  |  |  |     / /      \   \    |   __|  |  |  |  |    
# # .----)   |   |  |____ |  `--'  '--. / /_  .----)   |   |  |____ |  `--'  '--. 
# # |_______/    |_______| \_____\_____\____| |_______/    |_______| \_____\_____\
                                                                               

import torch
import torch.nn as nn
import math

class TransformerSeq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_dim, n_heads, hidden_dim, num_layers, dropout, device, max_len=100):
        super().__init__()
        self.device = device
        self.src_embedding = nn.Embedding(input_dim, embedding_dim)
        self.trg_embedding = nn.Embedding(output_dim, embedding_dim)

        self.pos_encoder = PositionalEncoding(embedding_dim, dropout, max_len)
        self.pos_decoder = PositionalEncoding(embedding_dim, dropout, max_len)

        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=n_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=False
        )

        self.fc_out = nn.Linear(embedding_dim, output_dim)

    def forward(self, src, trg):
        # src = [src_len, batch_size]
        # trg = [trg_len, batch_size]

        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.src_embedding.embedding_dim))
        trg_emb = self.pos_decoder(self.trg_embedding(trg) * math.sqrt(self.trg_embedding.embedding_dim))

        src_mask = None
        tgt_mask = self.generate_square_subsequent_mask(trg.size(0)).to(self.device)

        output = self.transformer(src_emb, trg_emb, src_mask=src_mask, tgt_mask=tgt_mask)
        return self.fc_out(output)

    def generate_square_subsequent_mask(self, sz):
        # Prevents attention to future positions in decoder
        return torch.triu(torch.ones((sz, sz)) * float('-inf'), diagonal=1)


# # .___________..______          ___       __  .__   __.  __  .__   __.   _______ 
# # |           ||   _  \        /   \     |  | |  \ |  | |  | |  \ |  |  /  _____|
# # `---|  |----`|  |_)  |      /  ^  \    |  | |   \|  | |  | |   \|  | |  |  __  
# #     |  |     |      /      /  /_\  \   |  | |  . `  | |  | |  . `  | |  | |_ | 
# #     |  |     |  |\  \----./  _____  \  |  | |  |\   | |  | |  |\   | |  |__| | 
# #     |__|     | _| `._____/__/     \__\ |__| |__| \__| |__| |__| \__|  \______| 
print(F"Printing the spectrum vocab length {len(spectrum_vocab)}")
print(F"Printing the selfie vocab length {len(selfie_vocab)}")


input_dim = len(spectrum_vocab)
output_dim = len(selfie_vocab)
encoder_embedding_dim = 256
decoder_embedding_dim = 256
hidden_dim = 512
n_layers = 2
encoder_dropout = 0.5
decoder_dropout = 0.5


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


model = TransformerSeq2Seq(
    input_dim=input_dim,
    output_dim=output_dim,
    embedding_dim=encoder_embedding_dim,
    n_heads=8,  # or your desired number of heads
    hidden_dim=hidden_dim,
    num_layers=n_layers,
    dropout=encoder_dropout,
    device=device,
    max_len=100
).to(device)


print("model succesfully put on device")

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


model.apply(init_weights)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f"The model has {count_parameters(model):,} trainable parameters")

optimizer = optim.Adam(model.parameters())

criterion = nn.CrossEntropyLoss(ignore_index=pad_index)

def train_fn(
    model, data_loader, optimizer, criterion, clip, teacher_forcing_ratio, device
):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(data_loader):
        # print("🔍 Keys in batch:", batch.keys())
        # print("📐 spectrum_ids shape:", batch["spectrum_ids"].shape)
        # print("📐 selfie_ids shape:", batch["selfie_ids"].shape)

        src = batch["spectrum_ids"].to(device)
        trg = batch["selfie_ids"].to(device)
        # src = [src length, batch size]
        # trg = [trg length, batch size]
        optimizer.zero_grad()
        output = model(src, trg)
        # output = [trg length, batch size, trg vocab size]
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        # output = [(trg length - 1) * batch size, trg vocab size]
        trg = trg[1:].view(-1)
        # trg = [(trg length - 1) * batch size]
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(data_loader)

def evaluate_fn(model, data_loader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            src = batch["spectrum_ids"].to(device)
            trg = batch["selfie_ids"].to(device)
            # src = [src length, batch size]
            # trg = [trg length, batch size]
            output = model(src, trg)  # turn off teacher forcing
            # output = [trg length, batch size, trg vocab size]
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            # output = [(trg length - 1) * batch size, trg vocab size]
            trg = trg[1:].view(-1)
            # trg = [(trg length - 1) * batch size]
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(data_loader)

n_epochs = 10
clip = 1.0
teacher_forcing_ratio = 0.5

best_valid_loss = float("inf")

for epoch in tqdm.tqdm(range(n_epochs)):
    train_loss = train_fn(
        model,
        train_data_loader,
        optimizer,
        criterion,
        clip,
        teacher_forcing_ratio,
        device,
    )
    valid_loss = evaluate_fn(
        model,
        valid_data_loader,
        criterion,
        device,
    )
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "tut1-model.pt")
    print(f"\tTrain Loss: {train_loss:7.3f} | Train PPL: {np.exp(train_loss):7.3f}")
    print(f"\tValid Loss: {valid_loss:7.3f} | Valid PPL: {np.exp(valid_loss):7.3f}")

    model.load_state_dict(torch.load("tut1-model.pt"))

model.load_state_dict(torch.load("tut1-model.pt"))
test_loss = evaluate_fn(model, test_data_loader, criterion, device)

print(f"| Test Loss: {test_loss:.3f} | Test PPL: {np.exp(test_loss):7.3f} |")




def translate_sentence(
    sentence,
    model,
    selfie_tokeniser,
    spectrum_tokeniser,
    spectrum_vocab,
    selfie_vocab,
    lower,
    sos_token,
    eos_token,
    device,
    max_output_length=25,
):
    model.eval()
    with torch.no_grad():
        # If the input is already a list or tensor of token indices:
        if isinstance(sentence, torch.Tensor):
            ids = sentence.tolist()
        elif isinstance(sentence, list) and all(isinstance(x, int) for x in sentence):
            ids = sentence
        else:
            # Assume it's a raw spectrum array
            tokens = spectrum_tokeniser.tokenise(sentence)
            tokens = [sos_token] + tokens + [eos_token]
            ids = spectrum_vocab.lookup_indices(tokens)

        tensor = torch.LongTensor(ids).unsqueeze(-1).to(device)

        hidden, cell = model.encoder(tensor)
        inputs = selfie_vocab.lookup_indices([sos_token])

        for _ in range(max_output_length):
            inputs_tensor = torch.LongTensor([inputs[-1]]).to(device)
            output, hidden, cell = model.decoder(inputs_tensor, hidden, cell)
            predicted_token = output.argmax(-1).item()
            inputs.append(predicted_token)
            if predicted_token == selfie_vocab[eos_token]:
                break

        tokens = selfie_vocab.lookup_tokens(inputs)
    return tokens


print(f"sentence 1: {test_data[1]['spectrum_ids']}")
print(f"sentence 16: {test_data[16]['spectrum_ids']}")

sentence = test_data[16]["spectrum_ids"]

expected_translation = test_data[16]["selfie_ids"]

expected_ids = expected_translation.tolist()
expected_tokens = selfie_tokeniser.inv_vocab
expected_token_strings = [selfie_tokeniser.inv_vocab[idx] for idx in expected_ids]
print(f"Expected translation: {expected_token_strings}")

print(sentence)
print(expected_translation)

translation = translate_sentence(
    sentence,
    model,
    selfie_tokeniser,
    spectrum_tokeniser,
    spectrum_vocab, 
    selfie_vocab,
    lower,
    sos_token,
    eos_token,
    device,
)


print(translation)

def tokens_to_selfie(tokens):
    expected_selfies = "".join(
    token for token in tokens if token not in ["<sos>", "<eos>", "<pad>"])
    return expected_selfies

expected_smiles = sf.decoder(tokens_to_selfie(expected_token_strings))

from rdkit import Chem
from rdkit.Chem import Draw

# Convert SMILES to RDKit molecule and draw
mol = Chem.MolFromSmiles(expected_smiles)
img = Draw.MolToImage(mol)
img.save("molecule_true.png")


translation_smiles = sf.decoder(tokens_to_selfie(translation))
mol = Chem.MolFromSmiles(translation_smiles)
img = Draw.MolToImage(mol)
img.save("molecule_output.png")


def plot_input_output(index):
    

    sentence = test_data[index]["spectrum_ids"]
    expected_translation = test_data[index]["selfie_ids"]

    expected_ids = expected_translation.tolist()
    expected_tokens = selfie_tokeniser.inv_vocab
    expected_token_strings = [selfie_tokeniser.inv_vocab[idx] for idx in expected_ids]
    print(f"Expected translation: {expected_token_strings}")

    print(sentence)
    print(expected_translation)

    translation = translate_sentence(
        sentence,
        model,
        selfie_tokeniser,
        spectrum_tokeniser,
        spectrum_vocab, 
        selfie_vocab,
        lower,
        sos_token,
        eos_token,
        device,
    )

    expected_smiles = sf.decoder(tokens_to_selfie(expected_token_strings))

    from rdkit import Chem
    from rdkit.Chem import Draw

    # Convert SMILES to RDKit molecule and draw
    mol = Chem.MolFromSmiles(expected_smiles)
    img = Draw.MolToImage(mol)
    img.save(f"molecule_true{index}.png")


    translation_smiles = sf.decoder(tokens_to_selfie(translation))
    mol = Chem.MolFromSmiles(translation_smiles)
    img = Draw.MolToImage(mol)
    img.save(f"molecule_output{index}.png")

    print(f"the translation for index {index} is {translation}")

for i in range(0,10):
    plot_input_output(i)