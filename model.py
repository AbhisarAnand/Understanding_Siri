import tensorflow as tf
from abc import ABC
from pathlib import Path
from tensorflow.keras.layers import Dropout, Dense
from transformers import BertTokenizer
from transformers import TFBertModel


class JointIntentAndSlotFillingModel(tf.keras.Model, ABC):

    def __init__(self, intent_num_labels=None, slot_num_labels=None,
                 dropout_prob=0.1):
        super().__init__(name="joint_intent_slot")
        self.model_name = "bert-base-cased"
        base_bert_model = TFBertModel.from_pretrained(self.model_name)
        self.bert = base_bert_model

        self.dropout = Dropout(dropout_prob)
        self.intent_classifier = Dense(intent_num_labels)
        self.slot_classifier = Dense(slot_num_labels)

    def call(self, inputs, **kwargs):
        tokens_output, pooled_output = self.bert(inputs, **kwargs, return_dict=False)
        tokens_output = self.dropout(tokens_output)
        slot_logits = self.slot_classifier(tokens_output)

        pooled_output = self.dropout(pooled_output)
        intent_logits = self.intent_classifier(pooled_output)

        return slot_logits, intent_logits


class NlpModel:

    def __init__(self):
        self.model_name = "bert-base-cased"
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.intent_names = Path("vocab.intent").read_text().split()
        self.intent_map = dict((label, idx) for idx, label in enumerate(self.intent_names))
        self.slot_names = ["[PAD]"] + Path("vocab.slot").read_text().strip().splitlines()
        self.slot_map = {}
        for label in self.slot_names:
            self.slot_map[label] = len(self.slot_map)
        self.model = JointIntentAndSlotFillingModel(intent_num_labels=len(self.intent_map),
                                                    slot_num_labels=len(self.slot_map))
        inputs = tf.constant(self.tokenizer.encode("Book a table for two at Le Ritz for Friday night!"))[None, :]
        _ = self.model(inputs)
        self.model.load_weights('output_weights.hdf5')

    def decode_predictions(self, p_text, intent_names, slot_names,
                           intent_id, slot_ids):
        info = {"intent": intent_names[intent_id]}
        collected_slots = {}
        active_slot_words = []
        active_slot_name = None
        for word in p_text.split():
            tokens = self.tokenizer.tokenize(word)
            current_word_slot_ids = slot_ids[:len(tokens)]
            slot_ids = slot_ids[len(tokens):]
            current_word_slot_name = slot_names[current_word_slot_ids[0]]
            if current_word_slot_name == "O":
                if active_slot_name:
                    collected_slots[active_slot_name] = " ".join(active_slot_words)
                    active_slot_words = []
                    active_slot_name = None
            else:
                new_slot_name = current_word_slot_name[2:]
                if active_slot_name is None:
                    active_slot_words.append(word)
                    active_slot_name = new_slot_name
                elif new_slot_name == active_slot_name:
                    active_slot_words.append(word)
                else:
                    collected_slots[active_slot_name] = " ".join(active_slot_words)
                    active_slot_words = [word]
                    active_slot_name = new_slot_name
        if active_slot_name:
            collected_slots[active_slot_name] = " ".join(active_slot_words)
        info["slots"] = collected_slots
        return info

    def nlu(self, p_text):
        intent_names, slot_names = self.intent_names, self.slot_names
        inputs = tf.constant(self.tokenizer.encode(p_text))[None, :]  # batch_size = 1
        outputs = self.model(inputs)
        slot_logits, intent_logits = outputs
        slot_ids = slot_logits.numpy().argmax(axis=-1)[0, 1:-1]
        intent_id = intent_logits.numpy().argmax(axis=-1)[0]

        return self.decode_predictions(p_text, intent_names, slot_names, intent_id, slot_ids)

    def return_bert_tokens(self, p_text):
        return self.tokenizer.tokenize(p_text)

    def show_predictions(self, p_text):
        intent_names, slot_names = self.intent_names, self.slot_names
        inputs = tf.constant(self.tokenizer.encode(p_text))[None, :]  # batch_size = 1
        outputs = self.model(inputs)
        slot_logits, intent_logits = outputs
        slot_ids = slot_logits.numpy().argmax(axis=-1)[0, 1:-1]
        intent_id = intent_logits.numpy().argmax(axis=-1)[0]
        intent_str = "## Intent: " + intent_names[intent_id]
        slots_str = "## Slots: \n"
        for token, slot_id in zip(self.tokenizer.tokenize(p_text), slot_ids):
            slots_str += "{}: {}\n".format(token, slot_names[slot_id])
        return intent_str, slots_str


if __name__ == '__main__':
    nlp_model_inst = NlpModel()
    text = "I would like to listen to Anima by Thom Yorke on Spotify"
    print(nlp_model_inst.return_bert_tokens(text))
    intent, slots = nlp_model_inst.show_predictions(text)
    print(intent)
    print(slots)
    print(nlp_model_inst.nlu(text))
