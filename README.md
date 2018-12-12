
# Projet d'Apprentissage Statistique 


## Papier utile
* UNSUPERVISED ADVERSARIAL IMAGE RECONSTRUCTION: https://openreview.net/pdf?id=BJg4Z3RqF7
notre papier
* DCGAN: https://arxiv.org/pdf/1511.06434.pdf
la base des GANs qui marche, le papier résume la plus part des tricks à utiliser pour stabiliser l'entraînement du générateur et du discriminateur.
* AMBIENTGAN: https://arxiv.org/pdf/1511.06434.pdf le papier présente l'un architecture de base pour le dé-bruitage de d'image. Notre papier améliore juste celui la
* SELF ATTENTION GENERATIVE ADVERSARIAL NETWORK: https://arxiv.org/pdf/1805.08318.pdf, c'est l'architecture du réseau utilisé dans notre papier. Ca permet au GAN de se "concentrer" sur des zones plus éloigné de l'image.s
 
##  Arborescence du repository
    ├── README.md
    ├── network             <- réseaux 
    |── data                <- chargement des données
    |── noise               <- module de bruit
    |── main                <- script pour l'entrînement