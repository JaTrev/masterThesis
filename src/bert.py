import transformers
import tensorflow as tf
import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForMaskedLM, AdamW, BertConfig
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
                      max_length: int = 256, batch_size: int = 16, testing_flag=False) -> [DataLoader]:

    input_ids = []
    attention_masks = []
    # sentences = []
    # sentences_tkns = []

    padding_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    # For every document...
    for doc in data:
        # if memory isn't big enough, split iteration into 2 using: [int(len(data) / 2)]
        sentences = doc.split(' .')

        processed_sentences = [s.strip().lower() for s in sentences if not len(s.split()) <= 3]

        for i_s, sentence in enumerate(processed_sentences):

            if len(sentence.split()) <= 3 or all([w not in vocab for w in sentence.split(' ')]):
                continue

            sentence += "."

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

            # test_sentences.append(sentence)
            # test_sentences_tkns.append(word_tokenize(sentence))

    lm_labels = [[t_id if t_id != padding_id else -100 for t_id in tensor[0]] for
                 tensor in input_ids]
    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    lm_labels = torch.tensor(lm_labels)

    if testing_flag:
        # Create the DataLoader.
        prediction_data = TensorDataset(input_ids, attention_masks, lm_labels)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_data_loader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
        return [prediction_data_loader]

    else:
        # Combine the training inputs into a TensorDataset.
        data_set = TensorDataset(input_ids, attention_masks, lm_labels)

        # 90-10 train-validation split.
        train_size = int(0.9 * len(data_set))
        val_size = len(data_set) - train_size

        train_data_set, val_data_set = random_split(data_set, [train_size, val_size])

        train_data_loader = DataLoader(train_data_set, sampler=RandomSampler(train_data_set), batch_size=batch_size)

        validation_data_loader = DataLoader(val_data_set,
                                            sampler=SequentialSampler(val_data_set),
                                            batch_size=batch_size)

        return [train_data_loader, validation_data_loader]


def train_bert(model: BertForMaskedLM, train_data_loader: DataLoader, validation_data_loader: DataLoader,
               optimizer: Optimizer = None,
               bert_model_type: str = "bert-base-uncased",
               epochs: int = 4, seed_val: int = 42):

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

            #   [0]: input ids, [1]: attention masks, [2]: lm labels
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


def calculate_bert_embeddings(model: BertForMaskedLM, prediction_dataloader: DataLoader, bert_layer: list = -1):
    embeddings = []

    # Put model in evaluation mode
    model.eval()

    for batch in prediction_dataloader:
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
        embeddings.extend([b for b in output[2][bert_layer]])

    print("Done calculating embeddings.")
    return embeddings


def calculate_bert_vocab_embeddings(embeddings: list, vocab: list,
                                    input_ids: list, tokenizer: transformers.BartTokenizer,
                                    vocab_weights_cutoff: int = 60):

    assert vocab_weights_cutoff > 1, "vocab_weights_cutoff must be larger than 1"

    assert len(embeddings) == len(input_ids), "embeddings and input_ids are not the same size"

    sentence_embeddings = []
    sentence_weights = []

    vocab_embeddings = [[] for _ in vocab]
    vocab_weights = [0 for _ in vocab]

    # go through every sentence
    for sent_index, sent_ids in enumerate(input_ids):

        sent_weights = 0
        sent_embedds = []

        # go through the entire sentence (all tokens)
        for w_index, w_id in enumerate(sent_ids):

            w_id = int(w_id)
            if w_id == 101:
                # skip token at the beginning of a sentence
                continue
            if w_id == 0 or w_id == 102:
                # at the end of the sentence
                break

            # add token embedding to temp sentence embedding list
            sent_embedds.append(
                embeddings[sent_index][w_index].cpu().numpy())

            # check if word is in the vocabulary
            word = tokenizer.ids_to_tokens[w_id]
            if word in vocab:

                vocab_index = vocab.index(word)
                vocab_embeddings[vocab_index].append(embeddings[sent_index][w_index].cpu().numpy())
                vocab_weights[vocab_index] += 1

                sent_weights += 1

        if sent_weights >= 2:
            # sentence is only relevant if it includes at least 2 vocab words

            sentence_embeddings.append(np.average(sent_embedds, axis=0).tolist())
            sentence_weights.append(sent_weights)

        else:
            # else ignore sentence
            continue

    new_vocab_list = []
    new_vocab_embeddings = []
    new_vocab_weights = []

    for i_v, v_embedding in vocab_embeddings:

        if vocab_weights[i_v] < vocab_weights_cutoff:
            # ignore insignificant vocabulary
            continue
        else:

            new_vocab_list.append(vocab[i_v])
            new_vocab_embeddings.append(np.average(v_embedding, axis=0).tolist())
            new_vocab_weights.append(vocab_weights[i_v])

    return new_vocab_list, new_vocab_embeddings, new_vocab_weights, sentence_embeddings, sentence_weights


if __name__ == "__main__":
    print("Testing...")
