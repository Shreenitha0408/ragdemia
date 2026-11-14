# Deploying RAGDemia

| Where | LLM | Who can upload | Persistence |
|---|---|---|---|
| **Your laptop** (`ragdemia ui`) | Ollama, local GPU | you | local disk |
| **Docker Compose** (`docker compose up`) | Ollama container | you | `./data` volume |
| **Public demo** (Hugging Face Space) | OpenAI-compatible API (Groq free tier) | visitors → private workspaces; you → shared library (admin password) | private HF dataset |

## Public mode in one paragraph

With `server.public: true`, every browser gets a random workspace id. A visitor's uploads are
parsed, embedded into an in-memory index that only their requests can search (the PDF file is
deleted straight after parsing), and forgotten after 6 h idle. Questions search the shared
library **and** the visitor's workspace together. Only a request carrying the admin password
(`X-Admin-Token`, compared in constant time) can change the shared library; those uploads are
mirrored to a private Hugging Face dataset so they survive the Space's ephemeral disk.
Questions and uploads are rate-limited per visitor IP.

## Hugging Face Space - step by step

1. **Accounts (free).** [huggingface.co](https://huggingface.co/join) and
   [console.groq.com](https://console.groq.com) → *API Keys* → create a key.
2. **Create the Space.** New Space → SDK **Docker** → *Blank* → hardware *CPU basic (free)*.
   Upload the two files from `deploy/huggingface/` (`Dockerfile`, `README.md`) to the Space.
3. **Settings → Variables and secrets.**

   | Kind | Name | Value |
   |---|---|---|
   | Secret | `GROQ_API_KEY` | your Groq key |
   | Secret | `RAGDEMIA_ADMIN_TOKEN` | a long password you choose (admin sign-in) |
   | Secret | `HF_TOKEN` | a Hugging Face token with **write** access |
   | Variable | `RAGDEMIA_SERVER__PERSIST_REPO` | `<your-username>/ragdemia-library` |

4. The Space builds (~10 min: PyTorch, models, sample corpus, index) and starts.
5. **Update after a code change:** push to GitHub → Space *Settings → Factory rebuild*.

### Changing the model or provider

Any OpenAI-compatible endpoint works by overriding three variables, e.g. OpenRouter:
`RAGDEMIA_LLM__BASE_URL=https://openrouter.ai/api/v1`, `RAGDEMIA_LLM__MODEL=<model id>`,
and a secret holding the key named by `RAGDEMIA_LLM__API_KEY_ENV`.

### Limits of the free tier (honest notes)

- *CPU basic* has 2 vCPUs: retrieval + reranking take ~2-5 s per question; generation ~1 s.
- Free Spaces sleep after 48 h without visitors; the first visit after that takes ~1 min.
- Groq's free tier is rate-limited; the per-visitor limit (8 requests/min) keeps one visitor
  from exhausting it.
