import transformers
import tensorflow as tf
import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForPreTraining, BertForMaskedLM, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.optim import Optimizer
from nltk.tokenize import word_tokenize
import datetime
import random
import numpy as np
import time
import os
import pickle

device_name = tf.test.gpu_device_name()
print("device_name: " + str(device_name))

# If there's a GPU available...
if torch.cuda.is_available():
    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

print("-----")
print('Hashseed is', os.environ.get("PYTHONHASHSEED"))
print("-----")


def format_time(elapsed):

    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def save_bert(model: transformers.BertModel, filename: str):
    model.save_pretrained(filename)


def load_bert(filename: str) -> transformers.BertModel:
    model = BertForMaskedLM.from_pretrained(
        filename,
        output_attentions=False,
        output_hidden_states=True
    )
    # Tell pytorch to run this model on the GPU since all torches are on GPU.
    model.cuda()
    return model


def load_bert_embeddings(filename: str) -> list:

    with open(filename, "rb") as f:
        return pickle.load(f)


def save_bert_embeddings(filename: str, bert_embeddings: list):

    with open(filename, "wb") as myFile:
        pickle.dump(bert_embeddings, myFile)


def init_bert_tokenizer(tokenizer_type: str = 'bert-base-uncased') -> transformers.BartTokenizer:
    return transformers.BertTokenizer.from_pretrained(tokenizer_type)


def bert_tokenization(tokenizer: transformers.BertTokenizer, data: list, vocab: list,
                      max_length: int = 256) -> [int]:

    input_ids = []
    attention_masks = []
    sentences = []
    sentences_tkns = []

    padding_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    # For every document...
    for doc in data:
        # if memory isn't big enough, split iteration into 2 using: [int(len(data) / 2)]
        sentences = doc.split(' .')

        processed_sentences = [s.strip().lower() for s in sentences if not len(s.split()) <= 3]

        for i_s, sentence in enumerate(processed_sentences):

            if len(sentence.split()) <= 3 or all([w not in vocab for w in sentence.split(' ')]):
                continue

            encoded_dict = tokenizer.encode_plus(
                sentence,
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt',
                truncation=True
            )

            # Add the encoded sentence to the list.
            input_ids.append(encoded_dict['input_ids'])

            # And its attention mask .
            attention_masks.append(encoded_dict['attention_mask'])

            sentences.append(sentence)
            sentences_tkns.append(word_tokenize(sentence))

    lm_labels = [[t_id if t_id != padding_id else -100 for t_id in tensor[0]] for tensor in input_ids]
    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    lm_labels = torch.tensor(lm_labels)

    return input_ids, attention_masks, lm_labels, sentences, sentences_tkns


def train_bert(input_ids, attention_masks, lm_labels,
               model: BertForMaskedLM = None,
               optimizer: Optimizer = None, batch_size: int = 16,
               bert_model_type: str = "bert-base-uncased",
               epochs: int = 4, seed_val: int = 42):

    # Combine the training inputs into a TensorDataset.
    data_set = TensorDataset(input_ids, attention_masks, lm_labels)

    # 90-10 train/validation split.
    train_size = int(0.9 * len(data_set))
    val_size = len(data_set) - train_size

    train_data_set, val_data_set = random_split(data_set, [train_size, val_size])

    train_data_loader = DataLoader(train_data_set, sampler=RandomSampler(train_data_set), batch_size=batch_size)

    validation_data_loader = DataLoader(val_data_set,
                                        sampler=SequentialSampler(val_data_set),
                                        batch_size=batch_size)

    if model is None:
        model = BertForMaskedLM.from_pretrained(
            bert_model_type,
            output_attentions=False,
            output_hidden_states=True
        )

    if optimizer is None:
        optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

    # Tell pytorch to run this model on the GPU.
    model.cuda()

    # Total number of training steps is [number of batches] x [number of epochs].
    total_steps = len(train_data_loader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    # Set the seed values
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    training_stats = []
    total_t0 = time.time()

    for epoch_i in range(0, epochs):

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        t0 = time.time()
        total_train_loss = 0
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_data_loader):

            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_data_loader), elapsed))

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()

            output = model(b_input_ids,
                           attention_mask=b_input_mask,
                           labels=b_labels
                           )
            # output:
            # loss = output[0]
            # logit = output[1]
            # hidden_states = output[2]
            loss = output[0]

            total_train_loss += loss.to('cpu').item()
            print("total_train_loss:" + str(total_train_loss))

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_data_loader)
        print("total_train_loss:" + str(avg_train_loss))

        training_time = format_time(time.time() - t0)

        print()
        print("  Average training loss: " + str(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

        # Validation #
        print()
        print("Running Validation...")

        t0 = time.time()
        model.eval()
        total_eval_loss = 0

        # Evaluate data for one epoch
        for batch in validation_data_loader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            with torch.no_grad():
                output = model(b_input_ids,
                               token_type_ids=None,
                               attention_mask=b_input_mask,
                               labels=b_labels)
            loss = output[0]

            # Accumulate the validation loss.
            total_eval_loss += loss.item()

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_data_loader)

        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: " + str(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print()
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

    return model


def calculate_bert_embeddings(model: BertForMaskedLM, data_loader: DataLoader = None,
                              input_ids=None, attention_masks = None, lm_labels = None,
                              bert_layer: int = -1, batch_size: int = 16, get_cls_token=False):

    assert data_loader is not None or (input_ids is not None and attention_masks is not None and lm_labels is not None)

    if data_loader is None:
        data = TensorDataset(input_ids, attention_masks, lm_labels)
        sampler = SequentialSampler(data)
        data_loader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    print("Fetching bert embeddings from layer: " + str(bert_layer))
    embeddings = []

    # Put model in evaluation mode
    model.eval()

    for batch in data_loader:
        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            output = model(b_input_ids,
                           token_type_ids=None,
                           attention_mask=b_input_mask,
                           labels=b_labels)

        # loss = output[0]
        # logit = output[1]
        # hidden_states = output[2], shape: [#batch, 256, 768]
        if get_cls_token:
            embeddings.extend([b for b in output[-2:][1][bert_layer][0]])
        else:
            embeddings.extend([b for b in output[2][bert_layer]])

    print("Done calculating embeddings.")
    return embeddings


def calculate_bert_vocab_embeddings(input_ids: list, embeddings: list, vocab: list, vocab_emb_dict: dict,
                                    tokenizer: transformers.BartTokenizer, get_cls_token:bool = False):
    assert len(embeddings) == len(input_ids), "embeddings and input_ids are not the same size"

    # go through every sentence
    for sent_index, sent_ids in enumerate(input_ids):

        # go through the entire sentence (all tokens)
        i = 0
        while i < 256:

            w_id = int(sent_ids[i].item())
            if w_id == 101:
                # skip token at the beginning of a sentence
                i += 1
                continue
            if w_id == 0 or w_id == 102:
                # at the end of the sentence
                break

            # check if word is in the vocabulary
            word = tokenizer.ids_to_tokens[w_id]

            if get_cls_token:
                w_embedding_list = [embeddings[sent_index].cpu().detach().numpy()]
            else:
                w_embedding_list = [embeddings[sent_index][i].cpu().detach().numpy()]
            # get all subwords
            while i < 255 and tokenizer.ids_to_tokens[sent_ids[i + 1].item()][:2] == "##":
                word = word + tokenizer.ids_to_tokens[sent_ids[i + 1].item()][2:]

                if get_cls_token:
                    w_embedding_list.append(embeddings[sent_index].cpu().detach().numpy())
                else:
                    w_embedding_list.append(embeddings[sent_index][i+1].cpu().detach().numpy())
                i += 1

            if word in vocab:

                if len(w_embedding_list) > 1:
                    # average over all substring embeddings
                    word_embedding = np.average(w_embedding_list, axis=0)
                else:
                    # must be 1 in the list
                    word_embedding = w_embedding_list[0]

                if word in vocab_emb_dict:
                    vocab_emb_dict[word].append(word_embedding)

                else:
                    vocab_emb_dict[word] = [word_embedding]
            i += 1

    return vocab_emb_dict


if __name__ == "__main__":
    print("Testing...")
