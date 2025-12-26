from huggingface_hub import run_uv_job, get_token

run_uv_job(
    "main.py",
    dependencies=[
        "torch",
        "sentence-transformers",
        "transformers",
        "peft",
        "datasets",
        "tqdm",
        "accelerate",
    ],
    env={"HF_TOKEN": get_token()},
    flavor="h100",
)