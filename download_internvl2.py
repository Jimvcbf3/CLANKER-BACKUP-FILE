# download_internvl2.py
from huggingface_hub import snapshot_download

def main():
    print("[*] Starting InternVL2-8B download (model weights only)…")
    # This will save in your HF cache: C:\Users\<YOU>\.cache\huggingface\hub
    local_dir = snapshot_download(
        repo_id="OpenGVLab/InternVL2-8B",
        local_dir="InternVL2-8B",        # folder in C:\ai-auto
        local_dir_use_symlinks=False,   # Windows fix
        resume_download=True            # resumes if interrupted
    )
    print(f"[+] Model downloaded to: {local_dir}")

if __name__ == "__main__":
    main()
