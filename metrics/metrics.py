import spacy
import warnings
import re
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Metrics:
    def __init__(self, name_model = "Unassigned"):
        self._nlp = spacy.load("es_core_news_md")
        self._name_model = name_model
        self._list_metrics = []
        self._metrics_data = {
            "filename":             None,
            "precision":            None,
            "recall":               None,
            "f1":                   None,
            "cosine_sim":           None,
            "levenshtein_distance": None,
            "labels":               None,
            "inv_levenshtein":      None,
            "overall":              None,
            "language":             None,
            "time":                 None
        }

    def set_filename(self, filename):
        self._metrics_data["filename"] = filename

    def embedding_similarity(self, str1, str2, threshold=0.8):
        doc1 = self._nlp(str1)
        doc2 = self._nlp(str2)
        similarity = doc1.similarity(doc2)
        return similarity >= threshold
    
    def erase_adverbs_determinants(self, texto):
        doc = self._nlp(texto)
        tokens_filtrados = [token.text for token in doc if token.pos_ not in ('ADP', 'DET')]
        return ' '.join(tokens_filtrados)
    
    def levenshtein_distance(self, s1, s2, show_progress=True):
        # Usar tqdm solo si show_progress es True
        iterable = tqdm(s1) if show_progress else s1

        if len(s1) < len(s2):
            s1, s2 = s2, s1
        if len(s2) == 0:
            self._metrics_data["levenshtein_distance"]  = len(s1)
            return self._metrics_data["levenshtein_distance"]

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(iterable):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        self._metrics_data["levenshtein_distance"] = previous_row[-1]

    def get_cos_sim(self, text_hoped, text_generated):
        vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w[\w\-/]*\b")
        tfidf_matrix = vectorizer.fit_transform([text_hoped, text_generated])
        try:
            cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            self._metrics_data["cosine_sim"] = cosine_sim[0][0]
        except: 
            self._metrics_data["cosine_sim"] = 0.0
        return self._metrics_data["cosine_sim"]

    def get_precison(self, true_positives, false_positives):
        self._metrics_data["precision"] = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

    def get_recall(self, true_positives, false_negatives):
        self._metrics_data["recall"] = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    def get_f1(self, precision, recall):
        self._metrics_data["f1"] = 2 * (precision * recall) / (precision + recall)

    # TODO: Mover a utils (?) 
    def create_distance_matrix(self, str1, str2):

        # Split the strings into words
        str1 = str1.split(' ')
        str2 = str2.split(' ')

        # Initialize the matrix
        m, n = len(str1), len(str2)
        matrix = np.zeros((m + 1, n + 1), dtype=int)

        # Fill the first row and first column
        for i in range(1, m + 1):
            matrix[i][0] = i
        for j in range(1, n + 1):
            matrix[0][j] = j

        # Fill the matrix with edit distances
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i - 1] == str2[j - 1]:
                    matrix[i][j] = matrix[i - 1][j - 1]

                else:
                    matrix[i][j] = 1 + min(matrix[i - 1][j], matrix[i][j - 1], matrix[i - 1][j - 1])

        # Create a DataFrame to include the words in the headers
        df = pd.DataFrame(matrix, index=[""] + str1, columns=[""] + str2)

        return df

    #TODO: Refactorizar
    def trace_path(self, matrix, str1, str2):
        i, j= len(str1), len(str2)

        deleted = []
        added = []
        unchanged = []

        while i > 0 or j > 0:
            if i > 0 and j > 0 and str1[i - 1] == str2[j - 1]:
                unchanged.append(str1[i - 1])
                i -= 1
                j -= 1

            elif i > 0 and (j == 0 or matrix.iloc[i, j] == matrix.iloc[i - 1, j] + 1):
                deleted.append(str1[i - 1])
                i -= 1

            elif j > 0 and (i == 0 or matrix.iloc[i, j] == matrix.iloc[i, j - 1] + 1):
                added.append(str2[j - 1])
                j -= 1
            else:
                deleted.append(str1[i - 1])
                added.append(str2[j - 1])
                i -= 1
                j -= 1

        return deleted[::-1], added[::-1], unchanged[::-1]  # Reverse the lists to have the correct order


    def calc_positives_and_negatives(self, list_of_ground_truth_labels, list_of_generated_labels):
        # Check if inputs are lists
        if not isinstance(list_of_ground_truth_labels, list) or not isinstance(list_of_generated_labels, list):
            raise TypeError("Both inputs must be lists")

        str1 = ' '.join(list_of_ground_truth_labels)
        str2 = ' '.join(list_of_generated_labels)
        # Create the distance matrix
        matrix = self.create_distance_matrix(str1, str2)

        # Trace the minimum path and get the operations
        added_words, deleted_words, unchanged_words = self.trace_path(matrix, str1.split(' '), str2.split(' '))
        FN = len(added_words)
        FP = len(deleted_words)
        TP = len(unchanged_words)
        return TP, FP, FN


    def calc_metrics(self, ground_truth, predictions):
        (   true_positives,
            false_positives,
            false_negatives
        ) = self.calc_positives_and_negatives(ground_truth, predictions)
        self.get_precison(true_positives, false_positives)
        self.get_recall(true_positives, false_negatives)
        self.get_f1(self._metrics_data["precision"], self._metrics_data["recall"])


    def evaluate(self, masked, generated):
        ground_truth = re.findall(r'\[\*\*(.*?)\*\*\]', masked)
        predictions = re.findall(r'\[\*\*(.*?)\*\*\]', generated)
        self._metrics_data["labels"] = [ground_truth, predictions]
        self.calc_metrics(ground_truth, predictions) # Calculates precision, recall, f1

    def calculate_inv_levenshtein(self):
        if int(self._metrics_data["levenshtein_distance"]) == 0:
            self._metrics_data["inv_levenshtein"] = 1
        else: 
            self._metrics_data["inv_levenshtein"] = (1/self._metrics_data["levenshtein_distance"])
    
    def calculate_overall(self):
        self._metrics_data["overall"] = self._metrics_data["precision"] + self._metrics_data["recall"] +  self._metrics_data["f1"] + self._metrics_data["cosine_sim"] + self._metrics_data["inv_levenshtein"]

    def calculate_time(self, start_time):
        self._metrics_data["time"] = time.time() - start_time

    def calculate(self, 
                  ground_truth, 
                  generated, 
                  classification_bool = True, 
                  cosine_sim_bool = True, 
                  levenshtein_bool = True,
                  time = False):
        '''
            That functions calculates all the metrics and stores them in the metrics_data variable
            Args:
                ground_truth: The ground truth labels
                generated: The generated labels
                classification_bool: Whether to calculate the classification metrics
                cosine_sim_bool: Whether to calculate the cosine similarity
                levenshtein_bool: Whether to calculate the levenshtein distance
            Returns:
                None
        '''
        if time: self.calculate_time(time)
        if classification_bool: self.evaluate(ground_truth, generated)
        if cosine_sim_bool: self.get_cos_sim(ground_truth, generated)
        if levenshtein_bool: self.levenshtein_distance(ground_truth, generated)
        if levenshtein_bool: self.calculate_inv_levenshtein()
        if classification_bool and levenshtein_bool and cosine_sim_bool: self.calculate_overall()
        self.add_language()

    def add_language(self):
        filename = self._metrics_data["filename"].split(".")[0]
        data = pd.read_csv('./data/carmen/language.tsv', sep='\t', index_col='filename')
        language = data.loc[filename, 'language']
        self._metrics_data["language"] = language

    def store_metrics(self):    
        self._list_metrics.append(self._metrics_data)
        self._metrics_data = {
            "filename":             None,
            "precision":            None,
            "recall":               None,
            "f1":                   None,
            "cosine_sim":           None,
            "levenshtein_distance": None,
            "labels":               None,
            "inv_levenshtein":      None,
            "overall":              None,
            "language":             None,
            "time":                 None
        }

    def get_metrics(self):
        return self._metrics_data

    def save_metrics(self):
        df = pd.DataFrame(self._list_metrics)
        df.to_csv(f"data/metrics/{self._name_model}_metrics.csv")