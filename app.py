import streamlit as st
from collections import defaultdict, Counter
import pytesseract
from PIL import Image   
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import re
from openai import OpenAI
import openai
import spacy
import plotly.express as px
import plotly.graph_objects as go
from statistics import mean
from spacy.cli import download
# Configurez l'API OpenAI
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
openai.api_key = OPENAI_API_KEY
st.session_state.OPENAI_API_KEY = OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY)

# # Charger le modèle spaCy pour le traitement NLP
# try:
nlp=spacy.load("fr_core_news_sm")
# except OSError:
#     # Si le modèle n'est pas installé, téléchargez-le
# download("fr_core_news_sm")
# nlp = nlp = spacy.load("./models/fr_core_news_sm")
# nlp=spacy.load()
# Fonction pour nettoyer et filtrer les mots-clés
def filter_keywords(text):
    """Filtre les mots-clés en supprimant les stop words et mots peu pertinents."""
    doc = nlp(text)
    filtered_keywords = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct and len(token.text) > 2]
    return filtered_keywords

# Fonction pour filtrer les images (formats valides et non w3.org)
def filter_images(images, base_url):
    """Filtre les images en excluant celles de w3.org et ne gardant que les formats valides."""
    valid_formats = ['.jpg', '.jpeg', '.png', '.webp']
    filtered_images = []

    for image in images:
        image_url = image["src"]
        if 'w3.org' not in image_url and any(image_url.endswith(ext) for ext in valid_formats):
            filtered_images.append({"src": urljoin(base_url, image_url), "alt": image.get("alt", "").strip()})
    
    return filtered_images

# Classe Page pour stocker et analyser les informations SEO de chaque page
class Page:
    def __init__(self, url, base_url):
        self.url = url
        self.base_url = base_url
        self.title = ""
        self.meta_description = ""
        self.h1 = ""
        self.text_content = ""
        self.images = []
        self.internal_links = set()
        self.seo_score = 0
        self.score_details = {}
        self.recommendations = {}
        self._fetch_and_parse()

    def _fetch_and_parse(self):
        """Récupère et analyse le contenu HTML de la page pour les éléments SEO."""
        try:
            response = requests.get(self.url, timeout=5)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            
            self.title = soup.title.string if soup.title else ""
            meta_desc = soup.find("meta", attrs={"name": "description"})
            self.meta_description = meta_desc["content"] if meta_desc else ""
            
            h1_tag = soup.find("h1")
            self.h1 = h1_tag.get_text(strip=True) if h1_tag else ""
            
            self.text_content = soup.get_text(separator=" ", strip=True)
            
            images = [
                {"src": urljoin(self.url, img["src"]), "alt": img.get("alt", "").strip()}
                for img in soup.find_all("img", src=True)
            ]
            self.images=filter_images(images,self.url)
            
            for a_tag in soup.find_all("a", href=True):
                link = self._is_valid_link(a_tag["href"])
                if link:
                    self.internal_links.add(link)
        except requests.RequestException:
            pass

    def _is_valid_link(self, href):
        link = urljoin(self.base_url, href)
        if link.startswith(self.base_url) and "amp" and "#" not in link.lower():
            return link
        return None

    def analyze_images_with_ocr(self):
        """Utilise PyTesseract pour vérifier la pertinence des images de la page."""
        main_keywords = re.findall(r'\b\w+\b', self.h1.lower())
        image_recommendations = []
        for image in self.images:
            try:
                image_data = requests.get(image["src"], stream=True).raw
                img = Image.open(image_data)
                ocr_text = pytesseract.image_to_string(img).lower()
                
                if any(keyword in ocr_text for keyword in main_keywords):
                    image_recommendations.append(f"{image['src']}, {image['alt']}. Ce que l'image montre : {ocr_text} ")
                else:
                    image_recommendations.append(f"❌ Image non pertinente : {image['src']} - Vérifier le contenu et le alt.")
            except Exception as e:
                image_recommendations.append(f"⚠️ Impossible d'analyser l'image {image['src']} - {e}")
        
        self.recommendations["Image Analysis"] = image_recommendations

    def calculate_seo_score(self):
        """Calcule un score SEO avec détails sur chaque critère."""
        score = 0
        self.score_details["Title Presence"] = 10 if self.title else 0
        self.score_details["Title Length"] = 5 if 10 <= len(self.title) <= 60 else 0
        self.score_details["Meta Description Presence"] = 10 if self.meta_description else 0
        self.score_details["Meta Description Length"] = 5 if len(self.meta_description) <= 160 and len(self.meta_description) > 0 else 0
        self.score_details["H1 Presence"] = 10 if self.h1 else 0
        self.score_details["Content Length"] = 10 if len(self.text_content.split()) >= 300 else 0
        self.score_details["Images Present"] = 10 if self.images else 0
        self.score_details["Internal Links Present"] = 10 if self.internal_links else 0
        self.seo_score = sum(self.score_details.values())

# Fonction pour explorer le site et stocker les pages
MAX_PAGES_TO_CRAWL = 2

def crawl_site(start_url, base_url):
    visited_links = set()
    all_pages = []
    links_to_explore = {start_url}
    pages_crawled = 0
    
    with st.spinner("Exploration des pages en cours..."):
        while links_to_explore and pages_crawled < MAX_PAGES_TO_CRAWL:
            next_level_links = set()
            for link in links_to_explore:
                if link not in visited_links:
                    visited_links.add(link)
                    page = Page(link, base_url)
                    page.calculate_seo_score()
                    page.analyze_images_with_ocr()
                    all_pages.append(page)
                    next_level_links.update(page.internal_links - visited_links)
                    pages_crawled += 1
            if next_level_links:
                links_to_explore = next_level_links
                time.sleep(1)
            else:
                break
    return all_pages

# Fonction pour obtenir des recommandations ChatGPT
def get_gpt_recommendations(page, main_keywords):

    # Extraire le texte OCR des images
    ocr_texts = []
    image_descriptions = []  # Liste pour stocker les descriptions des images

    for image in page.images:
        try:
            # Supposons que nous avons une méthode pour récupérer le texte OCR de chaque image
            image_data = requests.get(image["src"], stream=True).raw
            img = Image.open(image_data)
            ocr_text = pytesseract.image_to_string(img).lower()  # Utilisation de pytesseract pour l'OCR
            ocr_texts.append(ocr_text)
            
            # Construction de la description de l'image
            alt_text = image.get("alt", "Aucun texte alternatif")
            image_desc = f"Image {image['src']} avec alt : {alt_text}, Pertinence (OCR) : {ocr_text}"
            image_descriptions.append(image_desc)
            
        except Exception as e:
            # Ajouter un message d'erreur si l'OCR échoue
            ocr_texts.append(f"⚠️ Impossible d'analyser l'image {image['src']} - {e}")
            image_descriptions.append(f"⚠️ Impossible d'analyser l'image {image['src']}")

    # Joindre les descriptions des images pour l'analyse finale
    image_analysis = "\n".join(image_descriptions)
    
    prompt = (
        f"Voici un extrait du contenu de la page : {page.text_content[:500]}... "
        f"Les mots-clés principaux identifiés sont : {', '.join(main_keywords)}. "
        f"Meta description actuelle : {page.meta_description}. "
        f"En tenant compte des bonnes pratiques SEO et de l'expérience utilisateur (UX), "
        f"fournissez des recommandations pour optimiser le SEO et maximiser l'engagement des lecteurs. "
        f"Identifiez les sections clés où les mots-clés pourraient être intégrés pour un meilleur impact et proposez une meta description optimisée. "
        f"Suivez ce format :\n\n"
        f"1. **Recommandations SEO** :\n"
        f"   - [Exemple : Augmentez la densité des mots-clés principaux ({', '.join(main_keywords[:3])}) dans les titres et sous-titres.]\n"
        f"   - [Exemple : Ajouter des titres parlant et percutant, tout en restant concit comme pour la métadescription.]\n"
        f"   - [Exemple : Choisissez plutôt une image sur le sujet de ... car elle est plus adaptée au contexte.]\n\n"
        f"2. **Meta Description proposée :**\n"
        f"   - [Meta description optimisée basée sur le sujet et les mots-clés principaux]"
        f"Pour la meta-description elle doit faire moins de 50 mots et répondre à la question la plus importante sur le sujet."
        f"3. **Analyse des Images :**\n"
        f"Voici les images analysées et leur pertinence :\n"
        f"{image_analysis}\n"
        f"S'assurer que les images sont pertinentes par rapport au sujet de la page. Si une image n'est pas en lien avec le contenu, il est recommandé de la remplacer par une image plus appropriée.\n"
        f"Vérifier l'optimisation des images (taille réduite sans perte de qualité) pour un temps de chargement rapide.\n"
        f"Recommander des améliorations si nécessaire : par exemple, ajouter un texte alternatif manquant ou remplacer une image inappropriée. S'il n'y a pas d'image, vous dites 'Pas d'images' \n\n"
        f"\n\nGardez la réponse concise et efficace."
    )

    messages = [
        {"role": "system", "content": "Vous êtes un expert SEO qui fournit des recommandations."},
        {"role": "user", "content": prompt}
    ]
    
    response = client.chat.completions.create(
        messages=messages,
        model="gpt-3.5-turbo",
        max_tokens=300
    )

    return response.choices[0].message.content.strip()

def generate_seo_summary(all_pages, keyword_counts):

    # Calcul des KPIs
    total_pages = len(all_pages)
    avg_word_count = mean(len(page.text_content) for page in all_pages)
    total_images = sum(len(page.images) for page in all_pages)
    images_without_alt = sum(1 for page in all_pages for img in page.images if not img["alt"])

    # Liens internes et externes
    total_internal_links = sum(len(page.internal_links) for page in all_pages)
    total_external_links = sum(
        len([link for link in page.internal_links if not link.startswith(page.base_url)]) for page in all_pages
    )
    
    # Longueur moyenne des titres et des meta descriptions
    avg_title_length = mean(len(page.title) for page in all_pages if page.title)
    avg_meta_desc_length = mean(len(page.meta_description) for page in all_pages if page.meta_description)
    
    # Obtenir les 10 mots-clés les plus fréquents
    top_keywords = [keyword for keyword, count in keyword_counts.most_common(10)]
    
    # Générer une synthèse SEO ciblée avec ChatGPT en fonction des mots-clés principaux et des KPIs globaux
    summary_prompt = (
        f"Vous êtes un expert SEO analysant un site web sur une thématique spécifique. "
        f"Les mots-clés principaux du site sont : {', '.join(top_keywords)}. "
        f"Voici quelques statistiques SEO importantes pour le site :\n"
        f"- Nombre de pages analysées : {total_pages}\n"
        f"- Nombre moyen de mots par page : {avg_word_count:.1f}\n"
        f"- Total d'images sur le site : {total_images}\n"
        f"- Nombre d'images sans texte alternatif (alt) : {images_without_alt}\n"
        f"- Nombre total de liens internes : {total_internal_links}\n"
        f"- Nombre total de liens externes : {total_external_links}\n"
        f"- Longueur moyenne des titres (balises <title>) : {avg_title_length:.1f} caractères\n"
        f"- Longueur moyenne des descriptions (balises <meta description>) : {avg_meta_desc_length:.1f} caractères\n\n"
        f"En fonction de ces KPIs et des mots-clés principaux, proposez des recommandations concrètes "
        f"pour améliorer le SEO du site. Priorisez les actions ayant un impact fort sur le classement SEO et "
        f"qui ajoutent de la valeur pour l'utilisateur. Fournissez une synthèse avec des conseils ciblés sur "
        f"la densité des mots-clés, l’optimisation des balises HTML, l'amélioration de l'expérience utilisateur, "
        f"et l'optimisation de contenu. Justifier vos propos avec les données concrètement. Soyez précis et concis dans votre réponse."
    )

    # Appel à l'API GPT pour obtenir les recommandations générales basées sur le contexte du site
    overall_gpt_recommendations = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "Vous êtes un expert SEO fournissant des recommandations pour améliorer le classement SEO d'un site web."},
            {"role": "user", "content": summary_prompt}
        ],
        model="gpt-3.5-turbo",
    ).choices[0].message.content.strip()
    
    return overall_gpt_recommendations

# Fonction pour afficher une jauge de couleur pour le score moyen
def display_average_score_gauge(average_score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=average_score,
        title={'text': "Score SEO moyen"},
        gauge={'axis': {'range': [0, 70]},
               'bar': {'color': "black"},
               'steps': [
                   {'range': [0, 20], 'color': "red"},
                   {'range': [20, 40], 'color': "orange"},
                   {'range': [40, 70], 'color': "green"}]}))
    st.plotly_chart(fig)

# Fonction pour créer un treemap des mots-clés
def display_keyword_treemap(keyword_counts):
    keywords, counts = zip(*keyword_counts.most_common(10))
    fig = px.treemap(
        names=keywords,
        parents=[""] * len(keywords),
        values=counts,
        title="Top mots-clés utilisés sur l'ensemble du site",
        color=counts,
        color_continuous_scale="Viridis",
    )
    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
    st.plotly_chart(fig)

# Fonction pour afficher les critères de notation dans la sidebar
def display_scoring_criteria():
    st.sidebar.header("Critères de Notation SEO")
    st.sidebar.write("Les critères ci-dessous influencent le score SEO pour chaque page :")
    
    criteria = {
        "Présence du Titre": "10 points si le titre est présent",
        "Longueur du Titre": "5 points si le titre contient entre 10 et 60 caractères",
        "Présence de la Meta Description": "10 points si la meta description est présente",
        "Longueur de la Meta Description": "5 points si la meta description fait 160 caractères ou moins",
        "Présence du H1": "10 points si un H1 est présent",
        "Longueur du Contenu": "10 points si le contenu contient au moins 300 mots",
        "Présence d'Images": "10 points si des images sont présentes",
        "Présence de Liens Internes": "10 points si des liens internes sont présents"
    }

    for critere, details in criteria.items():
        st.sidebar.write(f"**{critere}** : {details}")


# Interface Streamlit
st.title("Analyse SEO Avancée de Site Web")

display_scoring_criteria() # Appel de la fonction pour afficher les critères dans la sidebar

url_input = st.text_input("Entrez l'URL de départ:")
if st.button("Analyser"):
    base_url = "{0.scheme}://{0.netloc}".format(urlparse(url_input))
    st.write(f"**URL de base détectée**: {base_url}")
    
    all_pages = crawl_site(url_input, base_url)
    
    sorted_pages = sorted(all_pages, key=lambda p: p.seo_score)

    # Calcul du score moyen
    average_score = sum(page.seo_score for page in all_pages) / len(all_pages) if all_pages else 0
    display_average_score_gauge(average_score)

    # Analyse OCR des images (on applique la méthode `analyze_images_with_ocr()` à chaque page)
    for page in all_pages:
        page.analyze_images_with_ocr()  # Appel de la méthode d'analyse OCR des images

    st.write("### Analyse globale des mots-clés")
    # Treemap des mots-clés
    all_text_content = " ".join(page.text_content for page in all_pages)
    all_keywords = filter_keywords(all_text_content)
    keyword_counts = Counter(all_keywords)
    display_keyword_treemap(keyword_counts)
    
    st.write("### Pages classées par score SEO (du plus bas au plus élevé)")
    for page in sorted_pages:
        color = f"rgba({255 - int(255 * page.seo_score / 70)}, {int(255 * page.seo_score / 70)}, 0, 0.5)"
        with st.expander(f"{page.url} - Score SEO : {page.seo_score}/70", expanded=False):
            st.markdown(
                    f"<div style='background-color:{color}; padding:10px;'>"
                    f"Titre : {page.title}<br>"
                    f"Meta Description : {page.meta_description}<br>"
                    f"H1 : {page.h1}<br>"
                    f"</div>", 
                    unsafe_allow_html=True
            )
        
            
            main_keywords = filter_keywords(page.text_content[:500])
            gpt_recommendation = get_gpt_recommendations(page, main_keywords)
            st.write(gpt_recommendation)


    all_text_content = " ".join(page.text_content for page in all_pages)
    all_keywords = filter_keywords(all_text_content)
    keyword_counts = Counter(all_keywords)
    
    overall_gpt_recommendations = generate_seo_summary(all_pages, keyword_counts)
    # Affichage du rapport final
    st.write("### Rapport SEO Global du Site")
    st.write(f"**Score SEO moyen** : {average_score:.2f}/70")
    st.write("### Recommandations générales pour le site")
    st.write(overall_gpt_recommendations)
