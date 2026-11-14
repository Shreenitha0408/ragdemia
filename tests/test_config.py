from ragdemia.config import load_config


def test_env_override(monkeypatch, tmp_path):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("retrieval:\n  top_k: 7\n")
    monkeypatch.setenv("RAGDEMIA_LLM__MODEL", "llama3.1:8b")
    monkeypatch.setenv("RAGDEMIA_RETRIEVAL__USE_RERANKER", "false")
    cfg = load_config(cfg_file)
    assert cfg.retrieval.top_k == 7
    assert cfg.llm.model == "llama3.1:8b"
    assert cfg.retrieval.use_reranker is False
