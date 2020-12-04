import numpy as np
from itertools import combinations
import razdel
from typing import Union, List, Tuple
import torch
from torch import nn
from transformers import BertTokenizer, BertModel
import networkx as nx


class Text2Summary:
    def __init__(self, bert_name="DeepPavlov/rubert-base-cased", device="cuda"):
        """Constructor.

        Args:
            bert_name (str): name of bert model from transformers (default 'rubert-base-cased').
            device ('cpu' of 'cuda'): torch device type (default 'cuda')
        """
        self.device = device
        self.bert = BertModel.from_pretrained(bert_name, output_hidden_states=True)
        self.bert.to(device)
        self.bert.eval()
        self.tokenizer = BertTokenizer.from_pretrained(bert_name)

    def _split_to_sents(self, text: str):
        sentences = [sent.text for sent in razdel.sentenize(text)]
        return sentences

    def _calc_embedding(
        self, text: List[str], return_numpy=False
    ) -> Union[torch.Tensor, np.ndarray]:
        """ Calculate sentence embedding. 

        Args:
            text (List[str]): list of sentences
            return_numpy (boolean): if True return numpy.ndarray, else torch.tensor 

        Returns:
            np.ndarray: sentence embedding with shape 'bert hidden_size (default: 768)'
        """

        sentences = [
            [token.text.lower() for token in razdel.tokenize(sent)] for sent in text
        ]

        encoded = self.tokenizer.batch_encode_plus(
            sentences,
            padding="longest",
            is_split_into_words=True,
            truncation="longest_first",
            max_length=256,
        )

        input_ids = torch.tensor(encoded["input_ids"]).to("cuda")
        attention_mask = torch.tensor(encoded["attention_mask"]).to("cuda")
        with torch.no_grad():
            hidden_states = self.bert(input_ids, attention_mask)["last_hidden_state"]
            sentence_embedding = hidden_states[:, 0, :]

        if return_numpy:
            return sentence_embedding.cpu().numpy()
        return sentence_embedding

    def _cosine_similarity(self, embeddings) -> np.ndarray:
        """ Calculate cosine similarity for given sentences. 

        Args:
            text (str)
        Returns:
            float: cosine similarity
        """

        eps = 1e-8
        embeds_n = embeddings.norm(dim=1)[:, None]
        embeds_norm = embeddings / torch.max(embeds_n, eps * torch.ones_like(embeds_n))
        sim_mt = torch.mm(embeds_norm, embeds_norm.transpose(0, 1))

        return sim_mt.cpu().numpy()

    def calc_page_rank(self, text: str) -> List[Tuple[int, str, str]]:
        """ Calculate page rank for sententec in text 

        Args:
            text (str)
        Returns:
            List[sentence number: int, 
                 score: str, 
                 sentence: str
                ]
        """
        sentences = self._split_to_sents(text)
        embeddings = self._calc_embedding(sentences)
        scores = self._cosine_similarity(embeddings)
        pairs = combinations(range(scores.shape[0]), 2)
        edges = [(i, j, scores[i, j]) for i, j in pairs]

        g = nx.Graph()
        g.add_weighted_edges_from(edges)

        page_rank = nx.pagerank(g)
        result = [
            (i, page_rank[i], sent)
            for i, sent in enumerate(sentences)
            if i in page_rank
        ]
        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def get_summary(self, text: str, summary_part=0.1):
        page_rank = self.calc_page_rank(text)
        # Get top sentences
        n_summary_sentences = max(int(len(page_rank) * summary_part), 1)
        result = page_rank[:n_summary_sentences]

        # Restore original sentence order
        result.sort(key=lambda x: x[0])

        # Restore summary text
        predicted_summary = " ".join([sentence for i, proba, sentence in result])
        predicted_summary = predicted_summary.lower()
        return predicted_summary

    def __call__(self, text: str):
        return self.get_summary(text)
