
# **Console**

## **Avant toute chose !** Installation des packages requis par python

Se placer à la racine du projet :
```bash
pip install -r requirements.txt
```

## Tester les imports
```bash
python3 main.py test
```

### Voir toutes les fonctions disponibles

```bash
python3 main.py --help
```

## Lancer les exercices

Exemple 1 : BE 1, exercice 2
```bash
python3 main.py run-be1 2
```

Exemple 2 : BE 2, exercice 3
```bash
python3 main.py run-be2 3
```

## **Remarque sur les schémas**
Lorsque le code génère des schémas ou des animations, pour passer à la suite,
il faut fermer la fenêtre. 

# **Code**

## Problèmes
Certaines fonctions sont trop couteuses en temps, de la façon dont je les ai implémenté, (typiquement l'exposant de Lyapunov pour l'attracteur de Lorentz, trop de points à traiter) j'ai donc affiché les résultats obtenus la première fois et commenté les lignes générant les résultats.

## Organisation des fichiers
```
.
├── BEs
│   ├── BE1.py
│   ├── BE2.py
│   ├── BE3.py
│   └── __init__.py
├── functions
│   ├── __init__.py
│   ├── attractors.py
│   ├── correlation_dim.py
│   ├── lyapunov.py
│   ├── plongement.py
│   └── utils.py
├── main.py
└── requirements.txt
├── README.md
```

### Fichiers dans BEs
Contiennent le déroulé des exercices, chaque fichier est alloué à un BE, les questions sont traitées ensemble dans une seule fonction.

### Fichiers functions
Contiennent les fonctions nécessaires au traitement des questions, comme la création des attracteurs, le calcul des exposants de lyapunov, etc.
Sont réparties par thématique.
