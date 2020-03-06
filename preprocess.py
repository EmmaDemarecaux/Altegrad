"""
# Preprocessing functions to be used for the different methods afterwards 
"""
import os
import codecs
from bs4 import BeautifulSoup
import re
from nltk.stem.snowball import FrenchStemmer
import collections


words_to_remove = ['aa', 'abonnement', 'abonnements', 'abonner', 'abonnez', 'abonné', 'abonnés', 'accueil', 'accède',
                   'accès', 'accéder', 'accédez', 'accédé', 'accédés', 'achat', 'achats', 'acheter', 'achetez',
                   'acheté', 'achetés', 'activer', 'activez', 'activé', 'activés', 'actualité', 'actualités', 'actus',
                   'adresse', 'adresses', 'agence', 'aide', 'aides', 'ailleurs', 'ainsi', 'ajouter', 'ajoutez',
                   'ajouté', 'ajoutés', 'aller', 'allez', 'allé', 'allés', 'alternate', 'ami', 'amis', 'annoncer',
                   'annuler', 'août', 'aperçu', 'application', 'applications', 'après', 'archive', 'archives',
                   'article', 'articles', 'astuce', 'astuces', 'auprès', 'auto', 'autre', 'avant', 'avantage',
                   'avantages', 'avantagespour', 'avis', 'avismême', 'avril', 'bangui', 'bas', 'besoin', 'besoins',
                   'bon', 'bonne', 'bonnes', 'bons', 'boulevard', 'boulevards', 'bureau', 'button', 'cadeau',
                   'cadeaux', 'caractères', 'carte', 'cartes', 'catalogue', 'cataloguesproduit', 'catégorie',
                   'catégories', 'cgu', 'cgv', 'champs', 'changer', 'changez', 'changé', 'changés', 'cher', 'cherche',
                   'chercher', 'cherchez', 'cherché', 'cherchés', 'chers', 'chez', 'ci', 'click', 'clicks', 'client',
                   'cliquant', 'cliquer', 'cliquez', 'cliqué', 'cliqués', 'code', 'codes', 'colis', 'collect',
                   'collecter', 'collectez', 'collecté', 'collectés', 'comment', 'commentaire', 'commentaires',
                   'commerciale', 'commerciales', 'commerciaux', 'communication', 'complète', 'compte', 'comptes',
                   'condition', 'conditions', 'connaître', 'connecter', 'connectez', 'connecté', 'connectés',
                   'conseillés', 'conseils', 'consommation', 'consommations', 'consultation', 'consultations',
                   'consulter', 'contact', 'contacter', 'contactez', 'contacté', 'contactés', 'contenu', 'contenus',
                   'cookies', 'coordonnées', 'copyright', 'créer', 'créez', 'créé', 'créés', 'days', 'dernier',
                   'derniers', 'dernière', 'dernières', 'description', 'descriptions', 'design', 'dessous',
                   'destockage', 'devez', 'diaporama', 'dimanche', 'donner', 'donnez', 'donné', 'données', 'droite',
                   'droits', 'dès', 'décembre', 'déconnecter', 'déconnectez', 'déconnecté', 'déconnectés', 'découvrir',
                   'déjà', 'détail', 'détails', 'détailsarticles', 'engagement', 'engagements', 'english', 'envoyer',
                   'envoyez', 'envoyé', 'envoyés', 'etc', 'event', 'events', 'eventsarchives', 'exclu', 'exclue',
                   'exclues', 'exclus', 'facebook', 'facilité', 'facilités', 'faire', 'faites', 'favoris', 'fermer',
                   'fermez', 'fermé', 'fermésbutton', 'fidélité', 'fidélités', 'fr', 'frais', 'français', 'français',
                   'française', 'french', 'futur', 'futurs', 'février', 'gauche', 'gif', 'google', 'group', 'guide',
                   'guides', 'général', 'générale', 'générales', 'gérer', 'gérez', 'géré', 'gérés', 'haut', 'home',
                   'html', 'ici', 'iframe', 'image', 'images', 'imprimer', 'information', 'informations', 'informer',
                   'informez', 'informé', 'informés', 'inscription', 'inscriptions', 'inscrire', 'inscrit', 'inscrite',
                   'inscrites', 'inscrits', 'inscriver', 'inscrivez', 'international', 'intranet', 'janvier', 'jeudi',
                   'jour', 'jours', 'jpeg', 'juin', 'jusqu', 'laisser', 'laissez', 'laissé', 'laissés', 'lien', 'liens',
                   'lieu', 'lifestyle', 'ligne', 'lire', 'liste', 'livraison', 'livraisons', 'livré', 'livrés', 'liés',
                   'logo', 'lundi', 'légal', 'légale', 'légales', 'légales', 'légaux', 'magazine', 'magazines', 'mai',
                   'mail', 'mails', 'main', 'mainsitetitle', 'mardi', 'marque', 'marques', 'mars', 'media', 'mention',
                   'mentions', 'mentions', 'menu', 'menus', 'merci', 'mercredi', 'minus', 'mise', 'mme', 'modification',
                   'modifications', 'modules', 'moins', 'mois', 'mot', 'mots', 'même', 'navigateur', 'navigation',
                   'newsletter', 'newsletters', 'next', 'nl', 'nom', 'noms', 'non', 'note', 'notes', 'nouveautés',
                   'novembre', 'nuit', 'nuits', 'numerik', 'octobre', 'offerte', 'offres', 'ok', 'opinion', 'opinions',
                   'oublier', 'oubliez', 'oublié', 'oubliés', 'ouvrer', 'ouvrez', 'ouvré', 'pack', 'page', 'page',
                   'pages', 'paiement', 'paiements', 'panier', 'paniers', 'paramètre', 'paramètres', 'paris', 'partir',
                   'passe', 'payez', 'personne', 'personnel', 'personnelle', 'personnelles', 'personnels', 'personnes',
                   'pexels', 'plan', 'plans', 'plus', 'plusplan', 'point', 'policy', 'populaire', 'populaires',
                   'postal', 'postaux', 'pouvez', 'pouvoir', 'pratique', 'pratiques', 'prev', 'principal', 'principale',
                   'privacy', 'privilégié', 'privilégiée', 'privilégiées', 'privilégiés', 'prix', 'pro', 'prochains',
                   'produits', 'programme', 'programmes', 'promo', 'promotion', 'promotionnel', 'promotionnelle',
                   'promotionnelles', 'promotionnels', 'promotions', 'propos', 'prénom', 'prénoms', 'publication',
                   'publications', 'quantité', 'quantités', 'question', 'questions', 'rapide', 'recevez', 'recevoir',
                   'recherche', 'rechercher', 'recherches', 'recherchez', 'recherché', 'recherchés', 'recrutement',
                   'recrutements', 'recruter', 'relais', 'relation', 'relations', 'rendez', 'renseignement',
                   'renseignements', 'responsabilité', 'responsabilités', 'restants', 'rester', 'restez', 'resté',
                   'restés', 'retourmarques', 'roularta', 'rubrique', 'rubriques', 'rue', 'rues', 'récent', 'récents',
                   'réf', 'répondons', 'réservation', 'réservations', 'réserver', 'réservez', 'réservé', 'réservés',
                   'samedi', 'section', 'semaine', 'semaines', 'septembre', 'service', 'service', 'services', 'site',
                   'sitecatégories', 'sites', 'soir', 'soirs', 'soirée', 'soirées', 'soldes', 'solutions', 'sorbonne',
                   'souhaitez', 'soumettre', 'submit', 'suivante', 'suivez', 'supplémentaires', 'svg', 'syndiquer',
                   'sécurisé', 'sécurisés', 'sérénité', 'tard', 'team', 'title', 'titre', 'titre', 'titres', 'toggle',
                   'total', 'tous', 'tous', 'tout', 'toute', 'toutes', 'trouver', 'trouvez', 'trouvé', 'trouvés',
                   'tweet', 'tweets', 'twitter', 'tél', 'télécharger', 'téléchargez', 'téléchargé', 'tôt', 'utile',
                   'utiles', 'utilespublications', 'utiliser', 'utilisez', 'utilisé', 'valable', 'vendredi', 'venir',
                   'vente', 'ville', 'voir', 'web', 'xml', 'zone', 'zones', 'économisez', 'éléments', 'évènement',
                   'évènements', 'être']


def clean_host_texts(data, tok, stpwds, punct, verbosity=5, remove_words=False):
    cleaned_data = []
    for counter, host_text in enumerate(data):
        # converting article to lowercase already done
        temp = BeautifulSoup(host_text, 'lxml')
        text = temp.get_text()  # removing HTML formatting
        text = text.replace('{html}', "")  # removing html
        text = re.sub(r'http\S+', '', text)  # removing url
        text = re.sub(r'\S*@\S*\s?', '', text)  # removing e-mail
        text = re.sub(r'\[\S*.\S*\]', '', text)  # removing any character between []
        text = re.sub(r'[^\s\.]+\.[^\s\.]+', '', text)  # removing any link 
        text = re.sub(r'[^\s\.]+\>', '', text)  # removing any string of the form string>
        text = text.translate(str.maketrans(punct, ' '*len(punct)))
        text = ''.join([l for l in text if not l.isdigit()])
        text = re.sub(' +', ' ', text)  # striping extra white space
        text = text.strip()  # striping leading and trailing white space
        tokens = tok.tokenize(text)  # tokenizing (splitting based on whitespace)
        if remove_words:
            wds_to_remove = stpwds + words_to_remove
        else:
            wds_to_remove = stpwds
        tokens = [token for token in tokens if (token not in wds_to_remove) and (len(token) > 1)]
        stemmer = FrenchStemmer()   
        tokens_stemmed = []
        for token in tokens:
            tokens_stemmed.append(stemmer.stem(token))
        tokens = tokens_stemmed
        cleaned_data.append(tokens)
        if counter % round(len(data)/verbosity) == 0:
            print(counter, '/', len(data), 'text cleaned')
    return [re.sub(r'\b(\w+)( \1\b)+', r'\1', ' '.join(l for l in sub_cleaned_data)) for 
            sub_cleaned_data in cleaned_data]


def remove_duplicates(train_file):
    with open(train_file, 'r') as f:
        train_data_ids = f.read().splitlines()

    train_data_ids = list(set(train_data_ids))
    duplicates_to_drop = [item for item, count in
                          collections.Counter([item.split(",")[0] for item
                                               in train_data_ids]).items() if count > 1]
    # Remove duplicates in training data: hosts + labels
    train_hosts = list()
    y_train = list()
    for row in train_data_ids:
        host, label = row.split(",")
        if host not in duplicates_to_drop:
            train_hosts.append(host)
            y_train.append(label.lower())
    return train_hosts, y_train 


def import_texts(texts_path):
    texts = dict()
    file_names = os.listdir(texts_path)
    for filename in file_names:
        with codecs.open(os.path.join(texts_path, filename), encoding='utf8', errors='ignore') as f:
            texts[filename] = f.read().replace('\n', '').lower()
    return texts
            
            
def generate_data(data_hosts, texts):
    """Get textual content of web hosts of the training set"""
    data = list()
    for host in data_hosts:
        if host in texts:
            data.append(texts[host])
        else:
            data.append('')  
    return data
