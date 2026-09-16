import Definitions

import os.path as osp
import joblib
import numpy as np

from src.DataPreprocessing import DataPreprocessing


class ModelController:

    def __init__(self):
        print("ModelController.__init__ ->")
        # Asegura en una variable la ruta de los modelos
        self.model_path = osp.join(Definitions.ROOT_DIR, "resources/models")

        # Rutas de cada artefacto del pipeline (TF-IDF -> SVD/LSA -> clasificador)
        self.tfidf_path = osp.join(self.model_path, "tfidf.joblib")
        self.svd_path = osp.join(self.model_path, "svd.joblib")
        self.model_path = osp.join(self.model_path, "model.joblib")

        # Carga de los artefactos entrenados
        self.tfidf = joblib.load(self.tfidf_path)
        self.svd = joblib.load(self.svd_path)
        self.model = joblib.load(self.model_path)

        # Clase de preprocesamiento de la información (limpieza de texto)
        self.d_processing = DataPreprocessing()

    def get_categories(self):
        print("ModelController.get_categories ->")
        return self.d_processing.get_categories()

    def predict(self, texto: str):
        """Recibe un texto libre y devuelve:
        - ods_pred: número de ODS predicho (1-17)
        - ods_name: nombre del ODS predicho
        - probabilidades: diccionario {numero_ods: probabilidad} ordenado desc.
        """
        print("ModelController.predict ->")

        # 1) Limpieza de texto (mismo preprocesamiento usado en el entrenamiento)
        texto_limpio = self.d_processing.transform(texto)

        # 2) Vectorización TF-IDF
        X_tfidf = self.tfidf.transform([texto_limpio])

        # 3) Reducción de dimensionalidad (LSA / TruncatedSVD)
        X_reduced = self.svd.transform(X_tfidf)

        # 4) Predicción y probabilidades calibradas
        y_pred = self.model.predict(X_reduced)[0]
        proba = self.model.predict_proba(X_reduced)[0]

        categorias = self.get_categories()
        probabilidades = {
            int(clase): float(p)
            for clase, p in sorted(
                zip(self.model.classes_, proba), key=lambda x: x[1], reverse=True
            )
        }

        ods_pred = int(y_pred)
        ods_name = categorias.get(ods_pred, "Desconocido")

        return ods_pred, ods_name, probabilidades
