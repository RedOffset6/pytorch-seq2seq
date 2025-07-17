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


 
#######
#
#    UNHASH TO TRAIN
#
##########

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
        if isinstance(sentence, torch.Tensor):
            ids = sentence.tolist()
        elif isinstance(sentence, list) and all(isinstance(x, int) for x in sentence):
            ids = sentence
        else:
            tokens = spectrum_tokeniser.tokenise(sentence)
            tokens = [sos_token] + tokens + [eos_token]
            ids = spectrum_vocab.lookup_indices(tokens)

        src = torch.LongTensor(ids).unsqueeze(1).to(device)  # [src_len, 1]
        trg_indexes = [selfie_vocab[sos_token]]

        for i in range(max_output_length):
            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(1).to(device)  # [cur_len, 1]
            output = model(src, trg_tensor)  # [cur_len, 1, vocab_size]
            next_token = output[-1, 0].argmax(-1).item()
            trg_indexes.append(next_token)
            if next_token == selfie_vocab[eos_token]:
                break

        tokens = selfie_vocab.lookup_tokens(trg_indexes)
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