# Vertex AI Setup

Carmina can call Gemini models through Google Cloud Vertex AI when `CLOUD_PROVIDER=vertex_ai`.

## Requirements

1. A Google Cloud project with Vertex AI enabled.
2. A service account with permission to call Vertex AI generative models.
3. A local service account JSON key, kept outside Git.
4. Environment variables configured in `.env`.

## Environment Variables

```env
CLOUD_PROVIDER=vertex_ai
MODELS=gemini-2.5-flash
VERTEX_PROJECT_ID=your-gcp-project-id
VERTEX_LOCATION=global
GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service-account-key.json
```

Use `VERTEX_LOCATION=global` for the newest preview models when required by Google Cloud. Older models may use regional locations such as `us-central1`.

## Important Security Notes

- Do not commit service account JSON files.
- Keep local credentials under `secrets/` or another ignored path.
- The repository `.gitignore` excludes `secrets/` and `*.json` to avoid accidental credential commits.

## Running

After configuring `.env`, run the benchmark normally:

```bash
python main.py
```

Outputs are written under `data/outputs/`, and logs follow the `LOG_DIR` value from `.env`.
