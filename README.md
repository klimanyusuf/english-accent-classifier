# English Accent Classifier (POC)

🎙️ Proof-of-concept tool to classify English accents from public video URLs using AI-based audio processing and Whisper embeddings.

## Why this project?

In this proof-of-concept, I aimed to create a simple but practical tool that could automate part of the hiring process by evaluating spoken English accents from video interviews. I chose to leverage Whisper for transcription and embeddings, with a simple classifier to demonstrate the viability of this approach.

## Approach & Limitations

- Uses Whisper to transcribe and validate English speech.
- Extracts Whisper encoder embeddings as a proxy for accent features.
- Applies a simple classifier to predict English accents.

**Limitations**:

- Mock classifier used for demonstration; real deployment requires training on real accent data.
- Whisper embeddings are not optimized for speaker accent — performance can be improved with speaker embedding models.
- Public video download depends on yt-dlp compatibility.

**Recommended datasets for real accent training**:

- [CommonVoice](https://commonvoice.mozilla.org/en/datasets)
- [VoxCeleb](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/)
- [AccentDB](https://github.com/ksingh7/AccentDB)
- [L2-Arctic](https://psi.engr.tamu.edu/l2-arctic-corpus/)

## How to run

1. Clone this repo.
2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the app:

    ```bash
    streamlit run app.py
    ```

4. Enter a video URL in the app and click "Run Accent Analysis" get the results.
5. video Url for testing: https://www.youtube.com/watch?v=2Xc9gXyf2G4

## How to replace mock classifier

To replace the current mock classifier:

1. Train a model on real accent embeddings.
2. Save your classifier as `classifier.pkl`:

    ```python
    with open("classifier.pkl", "wb") as f:
        pickle.dump(your_trained_classifier, f)
    ```

3. No changes required to `pipeline.py`.

---

Author: Khalid Liman Yusuf


