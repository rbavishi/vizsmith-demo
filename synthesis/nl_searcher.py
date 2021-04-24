import atexit
import shutil
import tempfile
from typing import Dict, List

import attr
import gensim
import tqdm
from gensim import corpora
from nltk import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

from synthesis.base_searcher import BaseSearcher
from synthesis.query import Query


@attr.s(cmp=False, repr=False)
class NaturalLanguageSearcher(BaseSearcher):
    """
    A synthesis engine based on the attributes seen during the code. Since API calls often use
    viz_functions must be a list of records (dicts) with each record having at least the following entries:
    1. 'code': The code corresponding to the viz_function.
    2. 'df_args': A dictionary from strings corresponding to the dataframe args to their individual metadata.
        The metadata must contain an entry for 'columns' mapped to a list of columns of the dataframe.
    3. 'col_args': A dictionary from strings corresponding to the column args to their individual metadata, if any.
    4. 'api_names': A collection of strings corresponding to the APIs used in the viz_function.
    5. 'nl': The natural language associated with the viz_function. Should be '' or None if not available.
    """

    viz_functions: List[Dict] = attr.ib()

    _nlp = attr.ib(init=False)
    _temp_dir = attr.ib(init=False)
    _dictionary = attr.ib(init=False)
    _tf_idf = attr.ib(init=False)
    _sim_matrix = attr.ib(init=False)

    def __attrs_post_init__(self):
        self._temp_dir = tempfile.mkdtemp()
        self.wnl = WordNetLemmatizer()
        self.build_index()

        atexit.register(NaturalLanguageSearcher._cleanup, self._temp_dir)

    def build_index(self):
        self.build_index_tfidf()

    def build_index_tfidf(self):
        nl_docs = [t['nl'] or '' for t in self.viz_functions]
        tokenized_docs = [self.tokenize_doc(doc) for doc in tqdm.tqdm(nl_docs)]
        dictionary = corpora.Dictionary(tokenized_docs)
        corpus = [dictionary.doc2bow(tok_doc) for tok_doc in tokenized_docs]
        tf_idf = gensim.models.TfidfModel(corpus)
        sims = gensim.similarities.Similarity(f"{self._temp_dir}/",
                                              tf_idf[corpus],
                                              num_features=len(dictionary))

        self._dictionary = dictionary
        self._tf_idf = tf_idf
        self._sim_matrix = sims

    def tokenize_doc(self, doc: str):
        words = word_tokenize(doc)
        pos_tagged_words = pos_tag(words)

        tokens = []
        for word, tag in pos_tagged_words:
            if word == ',' or word == '.':
                continue

            if tag.startswith("NN"):
                tokens.append(self.wnl.lemmatize(word, pos='n'))
            elif tag.startswith('VB'):
                tokens.append(self.wnl.lemmatize(word, pos='v'))
            elif tag.startswith('JJ'):
                tokens.append(self.wnl.lemmatize(word, pos='a'))
            else:
                tokens.append(word)

        tokens = [i.lower() for i in tokens]
        return tokens

    def search(self, query: Query) -> List[Dict]:
        query_str = query.query_str
        num_dfs = len(query.provided_dfs)
        num_cols = len(query.requested_cols)

        picked_viz_functions = self.search_tfidf(query_str)
        return [t for t in picked_viz_functions
                if len(t['df_args']) == num_dfs and len(t['col_args']) == num_cols]

    def search_tfidf(self, query_str: str):
        tokens = self.tokenize_doc(query_str)
        query_bow = self._dictionary.doc2bow(tokens)
        query_tf_idf = self._tf_idf[query_bow]
        viz_function_idxes = self._sim_matrix[query_tf_idf]
        scored_viz_functions = sorted(zip(viz_function_idxes, self.viz_functions),
                                  key=lambda x: -x[0])

        return [viz_function for score, viz_function in scored_viz_functions if score > 0]

    @staticmethod
    def _cleanup(path):
        shutil.rmtree(path, ignore_errors=True)

    def __del__(self):
        self._cleanup(self._temp_dir)
