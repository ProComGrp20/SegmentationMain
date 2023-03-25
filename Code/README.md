# Introduction 

En France, environ 3000-4000 prothèses de l’articulation trapézo-métacarpienne sont posées chaque année et ce nombre a tendance à augmenter en raison du vieillissement de la population et des avancées technologiques. L’objectif de ce projet est d’optimiser la phase pré-opératoire d’une chirurgie de l’articulation trapézo-métacarpienne et ainsi d’anticiper la variation anatomique inter-individuelle en proposant une prothèse personnalisée aux patients atteints d’arthrose.
Pour ce faire, nous avons mis au point un algorithme de segmentation automatique des os du métacarpe à partir d’images scanner (CT), se basant sur des méthodes de deep learning. Cela nous a permis ensuite de pouvoir imprimer en 3D des structures osseuses d’intérêt. Ce projet se décompose donc en trois axes principaux pour sa mise en oeuvre :
1. La première étape consiste en la segmentation manuelle de 15 scanners de main et du poignet (6150 images) qui serviront de données d’entraînement à notre algorithme.
2. La deuxième étape est l’entraînement de notre algorithme de segmentation automatique, basé sur une architecture U-Net. Cela nous permet d’obtenir une segmentation des os du carpe pour n’importe quel scanner.
3. Enfin, la dernière étape est le maillage des os segmentés, ainsi que leur impression 3D

# Contenu

Ce github contient le code permettant d'effectuer la segmentation automatique. Il contient également le poster résumant le projet ainsi que la méthode permettant d'effectuer une segmentation manuelle pour préparer les données.
