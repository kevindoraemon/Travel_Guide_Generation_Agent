import unittest
from dataclasses import replace

from qdrant_client import QdrantClient

from travel_planner.rag.config import get_rag_config
from travel_planner.rag.embeddings import LocalHybridEmbeddings
from travel_planner.rag.qdrant_store import QdrantTravelStore
from travel_planner.rag.schemas import TravelSourceDocument


class FakeEmbeddings:
    def __init__(self, config):
        self.sparse = LocalHybridEmbeddings(config)

    @staticmethod
    def _dense(text):
        return [
            float("北京" in text or "故宫" in text),
            float("上海" in text or "外滩" in text),
            float("美食" in text or "餐" in text),
        ]

    def embed_passages(self, texts):
        return [self._dense(text) for text in texts]

    def embed_query(self, text):
        return self._dense(text)

    def embed_sparse(self, text, *, query=False):
        return self.sparse.embed_sparse(text, query=query)


class QdrantStoreTest(unittest.TestCase):
    def setUp(self):
        self.config = replace(
            get_rag_config(),
            collection_name="test_travel_knowledge",
            dense_dimension=3,
            chunk_size=200,
            chunk_overlap=20,
        )
        self.client = QdrantClient(":memory:")
        self.store = QdrantTravelStore(
            self.config,
            client=self.client,
            embeddings=FakeEmbeddings(self.config),
        )
        self.store.ensure_collection()

    def tearDown(self):
        self.client.close()

    def test_dense_sparse_payload_filters_and_upsert(self):
        self.store.upsert_document(
            TravelSourceDocument(
                title="北京亲子五日路书",
                content="故宫需要预约。适合家庭出游。第二天游览天坛。",
                city="北京",
                language="zh",
            )
        )
        self.store.upsert_document(
            TravelSourceDocument(
                title="上海美食路线",
                content="外滩散步后安排本帮菜餐厅。",
                city="上海",
                language="zh",
            )
        )

        dense = self.store.search_dense("北京故宫", filters={"city": "北京"}, limit=20)
        sparse = self.store.search_sparse("故宫预约", filters={"topic": "family"}, limit=20)
        self.assertEqual(1, len(dense))
        self.assertEqual("北京", dense[0]["city"])
        self.assertEqual(1, len(sparse))
        self.assertEqual("family", sparse[0]["topic"])
        self.assertEqual(2, self.store.stats()["points"])


if __name__ == "__main__":
    unittest.main()
