from src.similarity import key_similarity, titleMatching
import pandas as pd
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLineEdit, QPushButton, QLabel, QListWidget)
from PyQt6.QtCore import Qt
import sys


class AppGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.df = pd.read_parquet('data/clean_movies.parquet', engine = 'pyarrow')

        self.data_enc = None
        self.faiss_index = None

        self.layout = QVBoxLayout()
        
        self.search_label = QLabel("Enter a Movie Title:")
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("e.g. Inception, Dune, Avengers...")
        
        self.search_btn = QPushButton("Find Similar Movies")
        self.search_btn.clicked.connect(self.handle_search)

        self.query_status = QLabel("")
        self.query_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.query_status.setStyleSheet("font-weight: bold; color: #0078d7;")

        self.results_label = QLabel("Recommendations:")
        self.results_list = QListWidget()

        self.layout.addWidget(self.search_label)
        self.layout.addWidget(self.search_input)
        self.layout.addWidget(self.search_btn)
        self.layout.addSpacing(10)
        self.layout.addWidget(self.query_status) 
        self.layout.addSpacing(10)
        self.layout.addWidget(self.results_label)
        self.layout.addWidget(self.results_list)

        self.setLayout(self.layout)
        self.setWindowTitle("AI Movie Recommender")
        self.resize(400, 500)
    
    def handle_search(self):
        query = titleMatching(self.df, self.search_input.text())
        if(query):
            self.load_assets()
            self.query_status.setText(f"Searching for matches to: {query}")
            sim_df = key_similarity(self.data_enc, self.faiss_index, self.df, query)
            self.results_list.clear()
            for title in sim_df['title']:
                self.results_list.addItem(title)
        else:
            self.query_status.setText("")
            self.results_list.clear()
    
    def load_assets(self):
        if self.faiss_index == None and self.data_enc == None:
            import numpy as np
            import faiss

            self.data_enc = np.load('data/encoded_data.npy')
            self.faiss_index = faiss.read_index('data/movie_index.faiss')
        



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AppGUI()
    window.show()
    sys.exit(app.exec())
