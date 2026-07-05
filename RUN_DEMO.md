# Run the presentation demo

Teammates: start here.

```powershell
# 1. Install deps (no venv required)
pip install -r presentation/requirements.txt

# 2. Check artifacts pulled from Git
python presentation/scripts/check_artifacts.py

# 3. Terminal 1
python presentation/api_server.py

# 4. Terminal 2
python -m http.server 8080

# 5. Browser
# http://localhost:8080/presentation/
```

Full setup, troubleshooting, and list of files that must be committed: **[presentation/README.md](presentation/README.md)**
