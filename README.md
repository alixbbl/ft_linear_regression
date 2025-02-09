# FT_LINEAR_REGRESSION

👉 POUR LANCER LE PROGRAMME TRAINING :
    'python -m programs.training ./data/data.csv'

👉 POUR LANCER LE PROGRAMME PREDICTION :
    'python -m programs.prediction ./data/data.csv'

## La fonction model(): 


## Les fonctions de couts : 

La MSE (Erreur Quadratique Moyenne) comme la MAE (Mean Absolute Error) sont des fonctions
de couts classiques, utilisees pour la mesure de l'erreur entre les prédictions et les vraies 
valeurs.

📌 Différences entre MSE et MAE

    ✅ MSE (Mean Squared Error) :
    
    Pénalise fortement les grandes erreurs (les erreurs élevées sont amplifiées)
    Lissage : Courbe plus lisse, facilite la descente de gradient
    Sensibilité aux outliers : ❌ Très sensible (les erreurs élevées explosent au carré)
    Optimisation : Plus facile avec la descente de gradient (dérivée continue)
    Utilisée pour : Régression linéaire classique

    ✅ MAE (Mean Absolute Error) :

    Pénalise toutes les erreurs de manière égale
    Lissage : Pas de courbe lisse (la dérivée n'est pas continue en 0)
    Sensibilité aux outliers : ✅ Plus robuste (les valeurs extrêmes ont moins d'impact)
    Optimisation : Plus difficile avec la descente de gradient (non-différentiable en 0)
    Utilisée pour : Modèles plus robustes aux outliers (ex : médiane, modèles avancés)

📌 Résumé rapide :

MSE = Plus simple à optimiser, mais sensible aux valeurs extrêmes.
MAE = Plus robuste, mais plus difficile à minimiser avec la descente de gradient.
🚀 Choix recommandé pour la régression linéaire classique : MSE.


## L'algorithme de minimisation de la fonction de cout : 
🛠️ Comment fonctionne l'algo dans les grandes lignes :

On fixe a 0 les theta0 et theta1.
On fixe a 1000 par exemple le nombre d'iterations maximales du calcul (on ne peut pas calculer a l'infini).
On determine un "learning rate", en general, on le fixe a 0.01.
Puis :
On fait une prédiction avec le modèle (ŷ = Xθ).
On mesure l’erreur avec la fonction de coût J(θ).
On calcule le gradient (∇J(θ)) pour savoir comment ajuster θ.
On met à jour θ avec θ = θ - α * gradient.
On répète jusqu’à ce que J(θ) soit minimisé.