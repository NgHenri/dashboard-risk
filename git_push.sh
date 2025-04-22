#!/bin/bash

# Vérifie si des changements existent
if [[ -z $(git status --porcelain) ]]; then
    echo "✅ Aucun changement à valider. Le répertoire est propre."
    exit 0
fi

# Demande un message de commit
echo "🔧 Entrez le message de commit :"
read commit_message

# Affichage du message saisi
echo "📝 Commit message : \"$commit_message\""

# Ajout des fichiers modifiés
git add .

# Création du commit
git commit -m "$commit_message"

# Push vers la branche main
git push origin main

# Message de confirmation
echo "✅ Push effectué avec succès sur la branche main."
