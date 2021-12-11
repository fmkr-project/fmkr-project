# Détection d'obstacles sur une voie ferrée
Dépôt de programmes pour la session 2021-2022 du TIPE.\
**La MCOT est disponible dans le dossier principal de ce dépôt en format .pdf et .odt**

## Motivation
Ayant pour envie de développer la partie "traitement de l'image" abordée en Informatique pour tous en deuxième année, et utilisant le logiciel "RailSim II" afin de concevoir des paysages ferroviaires réalistes, j'ai voulu étudier un peu plus en profondeur ce domaine de l'IPT qui n'est en Spé qu'effleuré afin d'allier ces deux programmes qui n'ont pourtant aucun lien apparent.  
  
Avec la diminution des passages à niveau gardés de nos jours, il devient crucial de protéger ces points critiques, en particulier depuis la recrudescence des suicides. Certes, les lignes LGV ne disposent pas de telles infrastructures, mais les lignes rurales montagneuses disposent parfois de PN difficiles d'accès pour les secours qui ne peuvent alors pas intervenir à temps pour sauver le blessé.  

## Objectifs du TIPE
- Pouvoir reconnaître sur une image le tracé de rails dans une configuration idéale (pas de trafic sur le passage à niveau, pas de train).
- Décider de la présence ou non d'un obstacle sur le passage à niveau.
- (si temps suffisant) Appliquer le programme obtenu sur un flux vidéo.
- (si temps suffisant et motivation...) Passer à un langage de programmation plus rapide comme C#.
  
## Structure du code
Dans un premier temps, une détection des contours faisant appel aux filtres de SOBEL, qu'on ne présente plus, est effectuée.\
La suite du code sera commentée en temps voulu...

## Prérequis au bon fonctionnement du code
Toutes les fonctions sont écrites en Python 3. La dernière version de Python 3 est donc nécessaire afin de les utiliser.\
Les modules numpy, matplotlib, imageio sont également utilisés.

## Sources et liens externes
Site du SCEI : https://www.scei-concours.fr \
Thread dédié sur le forum de la MP de Clemenceau : http://philippe.skler.free.fr/forum/viewtopic.php?f=19&t=363


