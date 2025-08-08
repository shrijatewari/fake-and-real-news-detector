Fake and Real News Detector

A machine learning powered web app built with Python and Streamlit that detects whether a given news headline or article is real or fake.

Features

* User-friendly interface built using Streamlit for quick and easy predictions
* Machine learning model trained on real and fake news datasets for accurate results
* Text preprocessing to clean and transform input text before prediction
* Displays how confident the model is in its decision

Project Structure
fake-and-real-news-detector/
│
├── app.py                - Main Streamlit application
├── model.pkl             - Trained ML model (joblib file)
├── vectorizer.pkl        - TF-IDF vectorizer
├── archive 2/            - Contains True.csv and Fake.csv datasets
├── requirements.txt      - Python dependencies
└── README.md             - Project documentation

Dataset
The project uses the True.csv and Fake.csv datasets containing thousands of real and fake news articles.

Dataset source: [https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)

Installation

1. Clone this repository
   git clone [https://github.com/shrijatewari/fake-and-real-news-detector.git](https://github.com/shrijatewari/fake-and-real-news-detector.git)
   cd fake-and-real-news-detector

2. Create a virtual environment
   python3 -m venv venv
   source venv/bin/activate   # Mac/Linux
   venv\Scripts\activate      # Windows

3. Install dependencies
   pip install -r requirements.txt

Usage
Run the Streamlit app:
streamlit run app.py

This will open a browser window where you can enter a news article or headline to check if it’s real or fake.

Requirements
The main dependencies are:

* streamlit – Web app framework
* scikit-learn – Machine learning algorithms
* pandas – Data manipulation
* joblib – Model loading and saving

Full list in requirements.txt

Future Improvements

* Integrate BERT or other transformer-based models for better accuracy
* Add fact-check API to cross-verify claims
* Support for multiple languages

Author
Shrija Tewari
GitHub: [https://github.com/shrijatewari](https://github.com/shrijatewari)

