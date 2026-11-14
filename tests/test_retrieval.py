from ragdemia.bm25 import BM25Index, tokenize
from ragdemia.retrieval import reciprocal_rank_fusion
from ragdemia.store import IndexStore


def test_bm25_prefers_rare_matching_terms():
    docs = ["the model uses bm25 for retrieval", "the model is a transformer", "cats and dogs"]
    idx = BM25Index().fit(docs)
    top = idx.search("bm25 retrieval", top_k=3)
    assert top[0][0] == 0
    assert all(i != 2 for i, _ in top)  # no overlap -> not scored at all


def test_bm25_roundtrip():
    idx = BM25Index().fit(["alpha beta", "beta gamma"])
    clone = BM25Index.from_dict(idx.to_dict())
    assert clone.search("gamma") == idx.search("gamma")


def test_tokenize_drops_stopwords():
    assert tokenize("What is the In-Batch negative?") == ["in-batch", "negative"]


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion({"a": [(1, 0.9), (2, 0.8)], "b": [(2, 12.0), (3, 9.0)]}, k=60)
    assert fused[0][0] == 2  # ranked by both retrievers
    assert fused[0][2]["a_rank"] == 2 and fused[0][2]["b_rank"] == 1


def test_hybrid_retrieval_finds_the_right_section(retriever):
    res = retriever.retrieve("What batch size and in-batch negatives were used?", top_k=3)
    assert res[0].chunk.section == "2 Method"
    assert [r.rank for r in res] == [1, 2, 3]


def test_store_save_load(tmp_path, store):
    store.save(tmp_path)
    loaded = IndexStore.load(tmp_path)
    assert len(loaded.chunks) == len(store.chunks)
    assert loaded.vectors.ntotal == store.vectors.ntotal
