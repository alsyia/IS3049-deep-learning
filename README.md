# IS3049-deep-learning

Ce repo contient notre projet d'électif deep learning (3A).

Le fichier compression-dimage-avec-autoencodeurs.pdf contient notre rapport.

Nous avons travaillé sur la compression d'images par autoencoeurs, et avons testé l'influence de différentes fonctions de coûts prenant en compte la sémantique et les textures de l'image.

Le réseau est implémenté en Keras (pour les parties hauts niveaux : définition globale de l'architecture) et en Tensorflow (pour les parties bas niveaux : surcharge de gradients, callbacks, couches particulières).

Pour lancer le programme, modifiez le fichier `ModelConfig.py` de façon à y indiquer le chemin de votre dataset et la taille des images. 

Ensuite :

```
pip install -r requirements.txt
python main.py
```

**/!\\** Pour faire tourner le modèle sur GPU, modifier les requirements en remplaçant `tensorflow` par `tensorflow-gpu`.
