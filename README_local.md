# Original CNN–BiLSTM–GAT interview demo

This package presents the original uploaded research model. It does not train a
new model or claim improved accuracy. All 246 model tensors are unchanged.
The supplied inference checkpoint omits optimizer state to reduce file size.

## Start locally

Use Python **3.11 or 3.12**, not Python 3.13 for these pinned dependencies.
On Windows you can double-click `run_demo.bat` after installing Python 3.12.
Alternatively, extract the ZIP before running the following commands from its folder.

Windows:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m streamlit run app.py
```

macOS / Linux:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

Open the local address printed in the terminal, usually http://localhost:8501.
The first install needs internet and may download large packages. Set up and
rehearse before your interview. After setup, this demo runs locally on CPU.
Allow at least several GB of free RAM and close heavy applications during use.

On a Linux host, OpenCV/MediaPipe may also require system libraries such as
libGL and libglib. A Docker option is included below.

## The interview sequence (about three minutes)

1. Open **Architecture**. Explain: "The CNN reads visual patterns. The BiLSTM
   processes their sequence. GAT processes facial landmark connections. The
   classifier combines both streams."
2. Open **Analyze a video** and upload an original short video with one clear
   face. Do not use a screen recording of the interface as an evaluation video.
3. Click **Analyze video**. Explain the predicted class, fake-class score and
   fixed 0.50 decision threshold. The score is not a calibrated guarantee.
4. Show **CNN Grad-CAM**. Say: "This shows visual regions influencing the
   model's predicted class. It is not a confirmed manipulation mask."
5. Show **frame sensitivity**. Say: "I replace one visual feature with the
   sequence mean and measure the absolute change in the class score. Graph
   features stay fixed. A peak does not prove the frame is manipulated."
6. Open **GAT attention diagnostic**. Explain that it exposes landmark-edge
   weights, not confirmed fake regions. It is computed at the highest
   sensitivity sample position, independently of the heatmap slider.
7. Open **Results & limitations**. Explain the checkpoint metadata and the
   remaining evaluation work. Download the figure/JSON if useful.

Suggested introduction: "This is the working research prototype from my
final-year project, combining CNN, BiLSTM and GAT for video deepfake detection."

## Recording and backup

The **Recorded walkthrough** tab contains your uploaded original recording.
Its old metric labels and explanation wording predate this review. It is
clearly labelled as playback, not live inference.

After testing with your original source video, record a new walkthrough of
this version using your computer's screen recorder. Include the architecture,
one inference result and the limitations. Keep that MP4 available separately
for an interview without internet. This package is a local application, not an
already-hosted public website.

## Evidence and limits

- Checkpoint metadata: validation accuracy **66.6667%**, F1 **71.1538%**;
  stored epoch field **5** (index convention not verified).
- These are stored training values, not independently reproduced evaluation.
- The old interface's 69.40% accuracy / 73.17% F1 on 180 videos has not been
  verified from a supplied evaluation report and is not displayed as verified.
- `checkpoint_record.json` contains hashes and the weight-preservation check.
- `runtime_check.json` describes only the checks actually completed here.
- The original training code/evaluation report were not supplied. The class
  mapping, landmark mapping and preprocessing match the uploaded demo code,
  but their equivalence to training still needs confirmation.
- Missing face detections are filled by repeated crops as in the original
  code, and the demo discloses this. Very short clips reuse sampled frames.
- GAT attention alone does not establish localization or added accuracy;
  those require suitable evaluation and controlled comparisons.
- No new training, architecture changes, or threshold optimization were done.

## Optional Docker run

```bash
docker build -t deepfake-interview .
docker run --rm -p 8501:8501 deepfake-interview
```

Open http://localhost:8501. The Docker recipe has not been built in the review
environment. Before any hosting, test the source video on the actual target.
