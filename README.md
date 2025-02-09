# FT_LINEAR_REGRESSION

👉 

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