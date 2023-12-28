# Speech Emotion Recognition Model

This repository contains a Speech Emotion Recognition (SER) model implemented in Python using the RAVDESS dataset. The model uses audio features such as Mel-Frequency Cepstral Coefficients (MFCC), Chroma, and Mel spectrograms to recognize emotions in speech.

## Project Structure

The project structure is as follows:

```
speech-emotion-recognition/
│
├── data/
│   ├── Actor_1/
│   ├── Actor_2/
│   ├── ...    
│   └── Actor_N/
│
├── modelForPrediction1.sav
├── SpeechEmotionRecognition.ipynb
└── README.md
```

- **data:** Contains the RAVDESS dataset, organized by actor folders.
- **modelForPrediction1.sav:** Saved trained model for emotion prediction.
- **SpeechEmotionRecognition.ipynb:** Jupyter Notebook containing the code for the Speech Emotion Recognition model.
- **README.md:** Documentation file providing an overview of the project, its structure, and instructions.

## Getting Started

### Prerequisites

Make sure you have the required Python libraries installed. You can install them using:

```bash
pip install librosa soundfile numpy scikit-learn pandas
```

### Running the Code

1. Open the `SpeechEmotionRecognition.ipynb` Jupyter Notebook.
2. Execute the notebook cells to load data, extract features, train the model, and evaluate its accuracy.
3. The trained model will be saved as `modelForPrediction1.sav` in the project directory.

## Usage

Once the model is trained and saved, you can use it for predicting emotions in new speech audio files. Load the saved model and use the `predict` function on extracted features.

```python
import pickle

# Load the saved model
filename = 'modelForPrediction1.sav'
loaded_model = pickle.load(open(filename, 'rb'))

# Extract features from a new speech audio file
new_feature = extract_feature("path/to/new/audio/file.wav", mfcc=True, chroma=True, mel=True)
new_feature = new_feature.reshape(1, -1)

# Predict emotion
prediction = loaded_model.predict(new_feature)
print("Predicted Emotion:", prediction[0])
```

## Results

The accuracy of the model on the test set is printed during the training process. Additionally, a DataFrame (`df`) is created to show actual vs predicted emotions for a sample of the test set.

## Authors

- Yogita Popat

## Acknowledgments

- RAVDESS dataset: [link](https://zenodo.org/record/1188976#.Yb_6j4gzbIU)

Feel free to contribute to the improvement of this Speech Emotion Recognition model!
