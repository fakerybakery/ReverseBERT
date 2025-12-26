# Reverse BERT (aka Reverse CLAP)

Can you go from text embeddings back to text? That's the experiment here.

The setup is pretty simple: take a sentence encoder (all-mpnet-base-v2, 768 dimensions) and freeze it. Then train a small projection layer that maps those embeddings into "soft prompt" tokens for a language model. The LLM learns to reconstruct the original text from just those projected embeddings.

It's far from perfect. You probably can't reconstruct the exact meaning of the text, but you can get the general idea/vibe of the original input.
