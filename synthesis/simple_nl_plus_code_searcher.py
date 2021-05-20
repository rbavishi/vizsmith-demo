import atexit
import itertools
import shutil
import tempfile
import ast
from typing import Optional, Dict, List

import attr
import astunparse
from whoosh.fields import TEXT, ID, Schema
from whoosh.index import create_in
from whoosh.qparser import QueryParser, OrGroup

from synthesis.base_searcher import BaseSearcher
from synthesis.query import Query
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
class SimpleNLPlusCodeSearcher(BaseSearcher):
    """
    A synthesis engine based on the attributes seen during the code. Since API calls often use
    viz_functions must be a list of records (dicts) with each record having at least the following entries:
    1. 'code': The code corresponding to the viz_function.
    2. 'df_args': A dictionary from strings corresponding to the dataframe args to their individual metadata, if any.
    3. 'col_args': A dictionary from strings corresponding to the column args to their individual metadata, if any.
    4. 'api_names': A collection of strings corresponding to the APIs used in the viz_function.
    """
    viz_functions: List[Dict] = attr.ib()

    _temp_dir = attr.ib(init=False)
    _ix = attr.ib(init=False)
    _nlp = attr.ib(init=False)
    _dictionary = attr.ib(init=False)
    _tf_idf = attr.ib(init=False)
    _sim_matrix = attr.ib(init=False)

    def __attrs_post_init__(self):
        self._temp_dir = tempfile.mkdtemp()
        self.wnl = WordNetLemmatizer()
        self.build_index()

        atexit.register(SimpleNLPlusCodeSearcher._cleanup, self._temp_dir)

    def build_index(self):
        self.build_index_tfidf()
        self.build_code_index()

    def build_code_index(self):
        schema = Schema(identifier=ID(stored=True), content=TEXT(stored=True))
        ix = create_in(self._temp_dir, schema)
        writer = ix.writer()

        for idx, viz_function in enumerate(self.viz_functions):
            writer.add_document(identifier=str(idx), content=self.get_document_text(viz_function))

        writer.commit()
        self._ix = ix

    def get_document_text(self, viz_function):
        processed_api_names = []
        for a_name in viz_function['api_names']:
            if '.' in a_name:
                a_name = a_name.split('.')[-1]

            processed_api_names.append(a_name)

        return " ".join(sorted(processed_api_names))

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

    def search(self, query: Query):
        nl_results = self.search_nl(query)
        code_results = self.search_code(query)

        seen = set()
        results = []
        for i in itertools.chain(code_results, nl_results):
            if id(i) not in seen:
                seen.add(id(i))
                results.append(i)

        return results

    def search_nl(self, query: Query) -> List[Dict]:
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

    def search_code(self, query: Query):
        query_str = query.query_str
        num_dfs = len(query.provided_dfs)

        with self._ix.searcher() as searcher:
            whoosh_query = QueryParser("content", self._ix.schema).parse(query_str)
            results = searcher.search(whoosh_query, limit=None)
            filtered = []
            for r in results:
                viz_function = self.viz_functions[int(r['identifier'])]
                if num_dfs is not None and len(viz_function['df_args']) != num_dfs:
                    continue

                filtered.append(viz_function)

            return filtered

    @staticmethod
    def _cleanup(path):
        shutil.rmtree(path, ignore_errors=True)

    def __del__(self):
        self._cleanup(self._temp_dir)


@attr.s(cmp=False, repr=False)
class WhooshNLPlusCodeSearcher(BaseSearcher):
    """
    A synthesis engine based on the attributes seen during the code. Since API calls often use
    viz_functions must be a list of records (dicts) with each record having at least the following entries:
    1. 'code': The code corresponding to the viz_function.
    2. 'df_args': A dictionary from strings corresponding to the dataframe args to their individual metadata, if any.
    3. 'col_args': A dictionary from strings corresponding to the column args to their individual metadata, if any.
    4. 'api_names': A collection of strings corresponding to the APIs used in the viz_function.
    """
    viz_functions: List[Dict] = attr.ib()

    _temp_dir = attr.ib(init=False)
    _ix = attr.ib(init=False)

    def __attrs_post_init__(self):
        self._temp_dir = tempfile.mkdtemp()
        self.build_index()

        atexit.register(WhooshNLPlusCodeSearcher._cleanup, self._temp_dir)

    def build_index(self):
        schema = Schema(identifier=ID(stored=True), content=TEXT(stored=True))
        ix = create_in(self._temp_dir, schema)
        writer = ix.writer()

        for idx, viz_function in enumerate(self.viz_functions):
            writer.add_document(identifier=str(idx), content=self.get_document_text(viz_function))

        writer.commit()
        self._ix = ix

    def get_document_text(self, viz_function):
        processed_code = self._strip_code(viz_function['code'])
        nl = viz_function['nl']
        return "\n".join([processed_code, nl])

    def search(self, query: Query):
        query_str = query.query_str
        num_dfs = len(query.provided_dfs)

        with self._ix.searcher() as searcher:
            og = OrGroup.factory(0.9)
            whoosh_query = QueryParser("content", self._ix.schema, group=og).parse(query_str)
            results = searcher.search(whoosh_query, limit=None)
            filtered = []
            for r in results:
                viz_function = self.viz_functions[int(r['identifier'])]
                if num_dfs is not None and len(viz_function['df_args']) != num_dfs:
                    continue

                filtered.append(viz_function)

            return filtered

    @staticmethod
    def _cleanup(path):
        shutil.rmtree(path, ignore_errors=True)

    def __del__(self):
        self._cleanup(self._temp_dir)

    def _strip_code(self, code: str):
        a = ast.parse(code).body[0]
        new_body = []
        for i in a.body:
            if isinstance(i, (ast.Import, ast.ImportFrom)):
                continue
            new_body.append(i)

        a.body = new_body[:]
        allowed = set()
        for n in ast.walk(a):
            if isinstance(n, ast.Call):
                allowed.add(n.func)

        for i in ast.walk(a):
            if isinstance(i, ast.Name) and i not in allowed:
                i.id = "_"

        return astunparse.unparse(a)
