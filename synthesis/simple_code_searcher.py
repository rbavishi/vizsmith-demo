import atexit
import shutil
import tempfile
from typing import Optional, Dict, List

import attr
from whoosh.fields import TEXT, ID, Schema
from whoosh.index import create_in
from whoosh.qparser import QueryParser

from synthesis.base_searcher import BaseSearcher
from synthesis.query import Query


@attr.s(cmp=False, repr=False)
class SimpleCodeSearcher(BaseSearcher):
    """
    A synthesis engine based on the attributes seen during the code. Since API calls often use
    viz_functions must be a list of records (dicts) with each record having at least the following entries:
    1. 'code': The code corresponding to the viz_function.
    2. 'df_args': A dictionary from strings corresponding to the dataframe args to their individual metadata, if any.
    3. 'col_args': A dictionary from strings corresponding to the column args to their individual metadata, if any.
    4. 'api_names': A collection of strings corresponding to the APIs used in the viz_function.
    """

    viz_functions: List[Dict] = attr.ib()

    _ix = attr.ib(init=False)
    _temp_dir = attr.ib(init=False)

    def __attrs_post_init__(self):
        self._temp_dir = tempfile.mkdtemp()
        self.build_index()

        atexit.register(SimpleCodeSearcher._cleanup, self._temp_dir)

    def build_index(self):
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

    def search(self, query: Query):
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
