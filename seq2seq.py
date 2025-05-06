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

# #defines a spectrum tokeniser class
# class SpectrumTokeniser:
#     def __init__(self):
#         self.sos_token = "<sos>"
#         self.eos_token = "<eos>"
    
#     def tokeniser(self, spectrum_array):
#         # Extract peak indices (where value == 1)
#         peak_indices = [str(idx) for idx, value in enumerate(spectrum_array) if value == 1]
#         return peak_indices
    
#     def tokenise_with_special_tokens(self, spectrum_array):
#         tokens = [self.sos_token] + self.tokeniser(spectrum_array) + [self.eos_token]
#         return tokens


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




# #initilises an instance of the tokeniser
# spectrum_tokeniser = SpectrumTokeniser()

# #tokenises a spectrum as an example
# example_spectrum = dataset["1000"]["binned_proton"]  # Get the binned spectrum
# tokenised_spectrum = spectrum_tokeniser.tokenise_with_special_tokens(example_spectrum)

# #prints the tokenised spectrum 
# print("PRINTING TOKENISED SPECTRUM")
# print(tokenised_spectrum)

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

# print(f"length of train data = {len(train_data['spectra'])}")
# print(f"test length = {len(test_data['spectra'])}")
# print(f"valid length = {len(valid_data['spectra'])}")


# dataset = datasets.load_dataset("bentrevett/multi30k")

# print(dataset)

# train_data, valid_data, test_data = (
#     dataset["train"],
#     dataset["validation"],
#     dataset["test"],
#)

# train_data[0]

# en_nlp = spacy.load("en_core_web_sm")
# de_nlp = spacy.load("de_core_news_sm")

# string = "What a lovely day it is today!"

# print([token.text for token in en_nlp.tokenizer(string)])

# def tokenize_example(example, en_nlp, de_nlp, max_length, lower, sos_token, eos_token):
#     en_tokens = [token.text for token in en_nlp.tokenizer(example["en"])][:max_length]
#     de_tokens = [token.text for token in de_nlp.tokenizer(example["de"])][:max_length]
#     if lower:
#         en_tokens = [token.lower() for token in en_tokens]
#         de_tokens = [token.lower() for token in de_tokens]
#     en_tokens = [sos_token] + en_tokens + [eos_token]
#     de_tokens = [sos_token] + de_tokens + [eos_token]
#     return {"en_tokens": en_tokens, "de_tokens": de_tokens}

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

# en_vocab = torchtext.vocab.build_vocab_from_iterator(
#     train_data["en_tokens"],
#     min_freq=min_freq,
#     specials=special_tokens,
# )

# de_vocab = torchtext.vocab.build_vocab_from_iterator(
#     train_data["de_tokens"],
#     min_freq=min_freq,
#     specials=special_tokens,
# )

# print(en_vocab.get_itos()[:10])

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

#print(train_data)

# train_data = train_data.map(numericalize_example, fn_kwargs=fn_kwargs)
# valid_data = valid_data.map(numericalize_example, fn_kwargs=fn_kwargs)
# test_data = test_data.map(numericalize_example, fn_kwargs=fn_kwargs)
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


class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src length, batch size]
        embedded = self.dropout(self.embedding(src))
        # embedded = [src length, batch size, embedding dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs = [src length, batch size, hidden dim * n directions]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # outputs are always from the top hidden layer
        return hidden, cell

# #  .----------------.  .----------------.  .----------------.  .----------------.  .----------------.  .----------------.  .----------------. 
# # | .--------------. || .--------------. || .--------------. || .--------------. || .--------------. || .--------------. || .--------------. |
# # | |  ________    | || |  _________   | || |     ______   | || |     ____     | || |  ________    | || |  _________   | || |  _______     | |
# # | | |_   ___ `.  | || | |_   ___  |  | || |   .' ___  |  | || |   .'    `.   | || | |_   ___ `.  | || | |_   ___  |  | || | |_   __ \    | |
# # | |   | |   `. \ | || |   | |_  \_|  | || |  / .'   \_|  | || |  /  .--.  \  | || |   | |   `. \ | || |   | |_  \_|  | || |   | |__) |   | |
# # | |   | |    | | | || |   |  _|  _   | || |  | |         | || |  | |    | |  | || |   | |    | | | || |   |  _|  _   | || |   |  __ /    | |
# # | |  _| |___.' / | || |  _| |___/ |  | || |  \ `.___.'\  | || |  \  `--'  /  | || |  _| |___.' / | || |  _| |___/ |  | || |  _| |  \ \_  | |
# # | | |________.'  | || | |_________|  | || |   `._____.'  | || |   `.____.'   | || | |________.'  | || | |_________|  | || | |____| |___| | |
# # | |              | || |              | || |              | || |              | || |              | || |              | || |              | |
# # | '--------------' || '--------------' || '--------------' || '--------------' || '--------------' || '--------------' || '--------------' |
# #  '----------------'  '----------------'  '----------------'  '----------------'  '----------------'  '----------------'  '----------------' 


class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hidden dim]
        # context = [n layers, batch size, hidden dim]
        input = input.unsqueeze(0)
        # input = [1, batch size]
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, embedding dim]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output = [seq length, batch size, hidden dim * n directions]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # seq length and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, hidden dim]
        # hidden = [n layers, batch size, hidden dim]
        # cell = [n layers, batch size, hidden dim]
        prediction = self.fc_out(output.squeeze(0))
        # prediction = [batch size, output dim]
        return prediction, hidden, cell
        
# #      _______. _______   ______      ___        _______. _______   ______      
# #     /       ||   ____| /  __  \    |__ \      /       ||   ____| /  __  \     
# #    |   (----`|  |__   |  |  |  |      ) |    |   (----`|  |__   |  |  |  |    
# #     \   \    |   __|  |  |  |  |     / /      \   \    |   __|  |  |  |  |    
# # .----)   |   |  |____ |  `--'  '--. / /_  .----)   |   |  |____ |  `--'  '--. 
# # |_______/    |_______| \_____\_____\____| |_______/    |_______| \_____\_____\
                                                                               

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert (
            encoder.hidden_dim == decoder.hidden_dim
        ), "Hidden dimensions of encoder and decoder must be equal!"
        assert (
            encoder.n_layers == decoder.n_layers
        ), "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio):
        # src = [src length, batch size]
        # trg = [trg length, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        batch_size = trg.shape[1]
        trg_length = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        # tensor to store decoder outputs
        outputs = torch.zeros(trg_length, batch_size, trg_vocab_size).to(self.device)
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # first input to the decoder is the <sos> tokens
        input = trg[0, :]
        # input = [batch size]
        # for t in range(1, trg_length):
        #     # insert input token embedding, previous hidden and previous cell states
        #     # receive output tensor (predictions) and new hidden and cell states
        #     output, hidden, cell = self.decoder(input, hidden, cell)
        #     # output = [batch size, output dim]
        #     # hidden = [n layers, batch size, hidden dim]
        #     # cell = [n layers, batch size, hidden dim]
        #     # place predictions in a tensor holding predictions for each token
        #     outputs[t] = output
        #     # decide if we are going to use teacher forcing or not
        #     teacher_force = random.random() < teacher_forcing_ratio
        #     # get the highest predicted token from our predictions
        #     top1 = output.argmax(1)
        #     # if teacher forcing, use actual next token as next input
        #     # if not, use predicted token
        #     input = trg[t] if teacher_force else top1
        #     # input = [batch size]


        for t in range(1, trg_length):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            top1 = top1.clamp(0, self.decoder.output_dim - 1)   # <--- ADD THIS LINE
            #input = trg[t] if teacher_force else top1
            if teacher_force:
                input = trg[t].clamp(0, self.decoder.output_dim - 1)
            else:
                input = top1


        return outputs


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


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


print("Helllo 1")
encoder = Encoder(
    input_dim,
    encoder_embedding_dim,
    hidden_dim,
    n_layers,
    encoder_dropout,
)
print("hello 2")
decoder = Decoder(
    output_dim,
    decoder_embedding_dim,
    hidden_dim,
    n_layers,
    decoder_dropout,
)
print("HELLO 4")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Number of trainable parameters: {count_parameters(encoder) + count_parameters(decoder):,}")

model = Seq2Seq(encoder, decoder, device).to(device)

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
        output = model(src, trg, teacher_forcing_ratio)
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
            output = model(src, trg, 0)  # turn off teacher forcing
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

# for epoch in tqdm.tqdm(range(n_epochs)):
#     train_loss = train_fn(
#         model,
#         train_data_loader,
#         optimizer,
#         criterion,
#         clip,
#         teacher_forcing_ratio,
#         device,
#     )
#     valid_loss = evaluate_fn(
#         model,
#         valid_data_loader,
#         criterion,
#         device,
#     )
#     if valid_loss < best_valid_loss:
#         best_valid_loss = valid_loss
#         torch.save(model.state_dict(), "tut1-model.pt")
#     print(f"\tTrain Loss: {train_loss:7.3f} | Train PPL: {np.exp(train_loss):7.3f}")
#     print(f"\tValid Loss: {valid_loss:7.3f} | Valid PPL: {np.exp(valid_loss):7.3f}")

#     model.load_state_dict(torch.load("tut1-model.pt"))

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
        # if isinstance(sentence, str):
        #     tokens = [token.text for token in spectrum_tokeniser(sentence)]
        # else:
        #     tokens = [token for token in sentence]
        # if lower:
        #     tokens = [token.lower() for token in tokens]
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




sentence = test_data[0]["spectrum_ids"]
expected_translation = test_data[0]["selfie_ids"]

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
img.save("molecule.png")