# Deploying RAGDemia

| Where | LLM | Who can upload | Persistence |
|---|---|---|---|
| **Your laptop** (`ragdemia ui`) | Ollama, local GPU | you | local disk |
| **Docker Compose** (`docker compose up`) | Ollama container | you | `./data` volume |
| **Public demo** (Oracle Cloud Always Free VM, or any Linux server) | OpenAI-compatible API (Groq free tier) | visitors → private workspaces; you → shared library (admin password) | server disk |
| Public demo on Hugging Face Spaces (needs HF PRO for Docker) | same | same | private HF dataset |

## Public mode in one paragraph

With `server.public: true`, every browser gets a random workspace id. A visitor's uploads are
parsed, embedded into an in-memory index that only their requests can search (the PDF file is
deleted straight after parsing), and forgotten after 6 h idle. Questions search the shared
library **and** the visitor's workspace together. Only a request carrying the admin password
(`X-Admin-Token`, compared in constant time) can change the shared library; those uploads are
stored on the server's disk (or, on Hugging Face Spaces, mirrored to a private dataset because the Space disk is ephemeral).
Questions and uploads are rate-limited per visitor IP.

## Oracle Cloud Always Free - step by step (recommended)

Oracle's *Always Free* tier includes an ARM server with up to 4 CPUs and 24 GB RAM at no cost,
permanently. A card is required to verify your identity; Always Free resources are not charged.

**A. Create the server (Oracle web console)**

1. Sign up at [oracle.com/cloud/free](https://www.oracle.com/cloud/free/). Pick a *home region*
   close to you; it cannot be changed later.
2. *Compute → Instances → Create instance*:
   - **Image:** Canonical Ubuntu 24.04
   - **Shape:** *Change shape → Ampere → VM.Standard.A1.Flex*, **2 OCPUs, 12 GB memory**
     (within Always Free). If you see "Out of capacity", try another *availability domain*
     or try again later.
   - **SSH keys:** *Generate a key pair for me* → **download the private key** (keep it safe).
   - **Create.**
3. Open ports 80 and 443: on the instance page click the **subnet** → **Default Security List**
   → *Add Ingress Rules* → Source CIDR `0.0.0.0/0`, IP protocol TCP, destination port `80,443`.
4. Copy the instance's **Public IP address**.

**B. Install RAGDemia (one command on the server)**

From your laptop (PowerShell), connect with the downloaded key:

```bash
ssh -i path\to\ssh-key.key ubuntu@<PUBLIC-IP>
```

Then, on the server:

```bash
curl -fsSL https://raw.githubusercontent.com/Shreenitha0408/ragdemia/main/deploy/oracle/install.sh -o install.sh
```

```bash
bash install.sh
```

It asks for your **Groq API key** and an **admin password** (typed on the server, never shared),
then builds and starts everything. After ~15 minutes the demo is live at
`https://<your-ip-with-dashes>.sslip.io`, with a free Let's Encrypt certificate obtained by Caddy.
[sslip.io](https://sslip.io) is a free DNS service that maps that name to your IP, so no domain
purchase is needed (you can enter your own domain instead).

**Updating** after new commits: SSH in and run `bash install.sh` again. Code is pulled and
rebuilt; documents, index and secrets are kept.

**Useful commands** (on the server, in `~/ragdemia/deploy/oracle`): `sudo docker compose logs -f app`
(live logs), `sudo docker compose restart app`, `sudo docker compose ps`.

## Hugging Face Space (requires a PRO subscription for Docker Spaces)

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
