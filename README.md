# Projet de Veille SEO et Analyse de Site Web
Ce projet consiste en un outil d'exploration et d'analyse SEO de sites web. Utilisant Streamlit, il permet de crawler un site web, d'analyser divers critères SEO (présence de balises, longueur de contenu, liens internes, images), et de proposer un score SEO pour chaque page explorée.

Fonctionnalités
Crawl de site web : Exploration de plusieurs pages en suivant les liens internes jusqu'à une limite définie.
Analyse SEO : Calcul d'un score SEO basé sur la présence et la qualité des balises (titre, meta description, H1), le nombre de mots, les liens internes, et les images.
OCR sur les images : Analyse de texte dans les images via OCR (pytesseract), permettant une meilleure évaluation de l'accessibilité.
Visualisation des résultats : Résumé des scores et détails des critères SEO pour chaque page, affichés dans une interface interactive Streamlit.
Prérequis

## 1. Environnement virtuel
Pour isoler les dépendances du projet, nous recommandons d'utiliser un environnement virtuel Python.
python -m venv env
source env/bin/activate  # Pour Linux/Mac
env\Scripts\activate  # Pour Windows

## 2. Installation des dépendances
Les bibliothèques nécessaires sont listées dans requirements.txt. Installez-les avec :
pip install -r requirements.txt
Remarque : Le modèle linguistique de spaCy (fr_core_news_sm) est inclus dans les dépendances et sera téléchargé automatiquement si pip a les autorisations nécessaires.

Utilisation
Lancer l'application
Pour démarrer l'application Streamlit, exécutez la commande suivante :
streamlit run app.py

## 3. Saisie des informations
Une fois l'application lancée, entrez l'URL de départ pour l'exploration et le domaine de base pour limiter le crawl aux liens internes. Cliquez sur "Lancer l'analyse".

## 4. Résultats
L'application affichera une liste des pages explorées, avec pour chaque page :
L'URL et le score SEO total
Le nombre de liens internes
La présence et la qualité des balises (titre, meta description, H1)
La longueur du contenu
Les images et le texte détecté par OCR

## Structure du Code
app.py : Le fichier principal pour l'application Streamlit, qui contient l'interface utilisateur.
Page : La classe principale représentant une page web avec ses attributs et ses méthodes d'analyse SEO.
Configuration SEO (Exemple)
Les principaux critères SEO :
criteria = {
    "Présence du Titre": "10%",
    "Longueur du Titre": "5%",
    "Présence de la Meta Description": "10%",
    "Longueur de la Meta Description": "5%",
    "Présence du H1": "10%",
    "Longueur du Contenu": "10%",
    "Présence d'Images": "10%",
    "Présence de Liens Internes": "10%"
}

## Technologies Utilisées
Python : Langage principal
Streamlit : Framework pour construire des applications web interactives
spaCy : Traitement du langage naturel, pour l'analyse linguistique
pytesseract : OCR pour l'analyse de texte dans les images
BeautifulSoup4 : Extraction de contenu HTML

## Limitations et Améliorations Futures :
Limite de Pages : L'exploration est limitée à un nombre défini de pages pour éviter la surcharge de ressources.
Analyse SEO Avancée : Les critères actuels peuvent être enrichis pour intégrer des éléments SEO plus avancés.
Amélioration de l'OCR : Ajuster l'analyse OCR pour mieux gérer les images complexes.
