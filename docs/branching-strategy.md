# Branching Strategy

This document defines the branching model for this project.  
The goal is to maintain clean and stable development workflows while supporting
fast experimentation and modular feature development.

---

# 1. Overview of Branch Structure

main
dev
feature/data
feature/model
feature/app
experiment/<name> (optional, temporary)

### main  
- Stable production-ready branch.  
- All completed features are merged here.  
- This branch should always remain clean and functional.  
- Tagged versions (v0.1, v0.2 ...) originate from here.

### dev  
- Integration branch for ongoing development.  
- All feature branches merge into this branch before going to `main`.  
- May contain work-in-progress code but should stay stable enough for daily work.

### feature/data  
- Data-related development: preprocessing, dataset loaders, augmentation, collators, etc.  
- Merges into `dev` after completion.

### feature/model  
- Model development branch:  
  - LLM fine-tuning  
  - Whisper / WavLM / TTS  
  - YOLO / CLIP / Diffusion  
  - LoRA experimentation  
  - Training pipelines  
- All training-related code is developed here.

### feature/app  
- Application-level development:  
  - Streamlit / Gradio UI  
  - FastAPI backend  
  - CLI tools  
  - inference pipelines  
- Responsible for connecting data + model into a usable application.

---

# 2. Workflow Summary

### New feature development
git checkout dev
git checkout -b feature/<name>
...
git add .
git commit -m "Add <feature>"
git push

### Merge flow
feature/* → dev → main

### Experiment flow
experiment/* → feature/model or dev (if successful)
experiment/* (delete) (if not used)

# 3. Rules

### 1. Never commit directly to `main`
All changes must come through a PR or merge from `dev`.

### 2. `dev` remains stable  
`dev` is the central integration point.  
Messy experiments go into `experiment/*` instead.

### 3. Small, atomic commits  
Clear commit messages improve traceability:
- `Add: dataset collator`
- `Fix: WavLM padding issue`
- `Refactor: inference pipeline`

### 4. Delete merged branches  
After merging, delete the remote branch to keep the repo clean.

---

# 4. Summary

This branching model offers:
- Clean separation of concerns (data / model / app)  
- A stable main & dev workflow  
- Freedom for experiments without polluting core branches  
- Scalable structure that can support solo or team development  

All branches should exist for a clear purpose and be merged or deleted when their lifecycle ends.