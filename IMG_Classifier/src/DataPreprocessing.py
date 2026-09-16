import re

from nltk.stem.snowball import SnowballStemmer

# Lista de "stopwords" en espanol, identica a nltk.corpus.stopwords.words('spanish').
# Se deja embebida como lista estatica -en lugar de usar nltk.download()- para que la
# aplicacion no dependa de descargar datos externos al arrancar en Streamlit Cloud.
SPANISH_STOPWORDS = set([
    "a", "al", "algo", "algunas", "algunos", "ante",
    "antes", "como", "con", "contra", "cual", "cuando",
    "de", "del", "desde", "donde", "durante", "e",
    "el", "ella", "ellas", "ellos", "en", "entre",
    "era", "erais", "eran", "eras", "eres", "es",
    "esa", "esas", "ese", "eso", "esos", "esta",
    "estaba", "estabais", "estaban", "estabas", "estad", "estada",
    "estadas", "estado", "estados", "estamos", "estando", "estar",
    "estaremos", "estará", "estarán", "estarás", "estaré", "estaréis",
    "estaría", "estaríais", "estaríamos", "estarían", "estarías", "estas",
    "este", "estemos", "esto", "estos", "estoy", "estuve",
    "estuviera", "estuvierais", "estuvieran", "estuvieras", "estuvieron", "estuviese",
    "estuvieseis", "estuviesen", "estuvieses", "estuvimos", "estuviste", "estuvisteis",
    "estuviéramos", "estuviésemos", "estuvo", "está", "estábamos", "estáis",
    "están", "estás", "esté", "estéis", "estén", "estés",
    "fue", "fuera", "fuerais", "fueran", "fueras", "fueron",
    "fuese", "fueseis", "fuesen", "fueses", "fui", "fuimos",
    "fuiste", "fuisteis", "fuéramos", "fuésemos", "ha", "habida",
    "habidas", "habido", "habidos", "habiendo", "habremos", "habrá",
    "habrán", "habrás", "habré", "habréis", "habría", "habríais",
    "habríamos", "habrían", "habrías", "habéis", "había", "habíais",
    "habíamos", "habían", "habías", "han", "has", "hasta",
    "hay", "haya", "hayamos", "hayan", "hayas", "hayáis",
    "he", "hemos", "hube", "hubiera", "hubierais", "hubieran",
    "hubieras", "hubieron", "hubiese", "hubieseis", "hubiesen", "hubieses",
    "hubimos", "hubiste", "hubisteis", "hubiéramos", "hubiésemos", "hubo",
    "la", "las", "le", "les", "lo", "los",
    "me", "mi", "mis", "mucho", "muchos", "muy",
    "más", "mí", "mía", "mías", "mío", "míos",
    "nada", "ni", "no", "nos", "nosotras", "nosotros",
    "nuestra", "nuestras", "nuestro", "nuestros", "o", "os",
    "otra", "otras", "otro", "otros", "para", "pero",
    "poco", "por", "porque", "que", "quien", "quienes",
    "qué", "se", "sea", "seamos", "sean", "seas",
    "sentid", "sentida", "sentidas", "sentido", "sentidos", "seremos",
    "será", "serán", "serás", "seré", "seréis", "sería",
    "seríais", "seríamos", "serían", "serías", "seáis", "siente",
    "sin", "sintiendo", "sobre", "sois", "somos", "son",
    "soy", "su", "sus", "suya", "suyas", "suyo",
    "suyos", "sí", "también", "tanto", "te", "tendremos",
    "tendrá", "tendrán", "tendrás", "tendré", "tendréis", "tendría",
    "tendríais", "tendríamos", "tendrían", "tendrías", "tened", "tenemos",
    "tenga", "tengamos", "tengan", "tengas", "tengo", "tengáis",
    "tenida", "tenidas", "tenido", "tenidos", "teniendo", "tenéis",
    "tenía", "teníais", "teníamos", "tenían", "tenías", "ti",
    "tiene", "tienen", "tienes", "todo", "todos", "tu",
    "tus", "tuve", "tuviera", "tuvierais", "tuvieran", "tuvieras",
    "tuvieron", "tuviese", "tuvieseis", "tuviesen", "tuvieses", "tuvimos",
    "tuviste", "tuvisteis", "tuviéramos", "tuviésemos", "tuvo", "tuya",
    "tuyas", "tuyo", "tuyos", "tú", "un", "una",
    "uno", "unos", "vosotras", "vosotros", "vuestra", "vuestras",
    "vuestro", "vuestros", "y", "ya", "yo", "él",
    "éramos",
])


class DataPreprocessing:
    """Encapsula la limpieza de texto usada durante el entrenamiento del modelo de ODS.

    Debe aplicarse EXACTAMENTE la misma transformacion que se uso al entrenar el
    TfidfVectorizer (resources/models/tfidf.joblib), para que el vocabulario y los
    pesos aprendidos sean consistentes con el texto de entrada del usuario.
    """

    def __init__(self):
        print("DataPreprocessing.__init__ ->")
        self.stemmer = SnowballStemmer("spanish")

    def transform(self, texto: str) -> str:
        """Limpia y normaliza un texto libre: minusculas, sin URLs/digitos/puntuacion,
        sin stopwords, con stemming (misma logica que en el notebook de entrenamiento).
        """
        print("DataPreprocessing.transform ->")
        texto = str(texto).lower()
        texto = re.sub(r"http\S+|www\.\S+", " ", texto)
        texto = re.sub(r"[^a-záéíóúñü\s]", " ", texto)
        tokens = texto.split()
        tokens = [
            self.stemmer.stem(t)
            for t in tokens
            if t not in SPANISH_STOPWORDS and len(t) > 2
        ]
        return " ".join(tokens)

    def get_categories(self):
        """Nombres oficiales de los Objetivos de Desarrollo Sostenible (ODS 1-17)."""
        print("DataPreprocessing.get_categories ->")
        return {
            1: "Fin de la pobreza",
            2: "Hambre cero",
            3: "Salud y bienestar",
            4: "Educacion de calidad",
            5: "Igualdad de genero",
            6: "Agua limpia y saneamiento",
            7: "Energia asequible y no contaminante",
            8: "Trabajo decente y crecimiento economico",
            9: "Industria, innovacion e infraestructura",
            10: "Reduccion de las desigualdades",
            11: "Ciudades y comunidades sostenibles",
            12: "Produccion y consumo responsables",
            13: "Accion por el clima",
            14: "Vida submarina",
            15: "Vida de ecosistemas terrestres",
            16: "Paz, justicia e instituciones solidas",
            17: "Alianzas para lograr los objetivos",
        }

    def get_cat_name(self, index: int) -> str:
        print("DataPreprocessing.get_cat_name ->")
        return self.get_categories().get(index, "Desconocido")
