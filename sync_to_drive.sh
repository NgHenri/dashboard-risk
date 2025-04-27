#!/bin/bash

# Nom du remote rclone (gdrive ici)
REMOTE_NAME="gdrive"

# Chemin distant sur Google Drive
REMOTE_PATH="Python/OCRP/Projet07/working/application_test/dashboard-risk"

# Chemin local du dossier à copier
LOCAL_PATH="./"  # ou remplace par le chemin absolu si nécessaire

# Commande de synchronisation
echo "📤 Synchronisation de $LOCAL_PATH vers $REMOTE_NAME:$REMOTE_PATH ..."
rclone copy "$LOCAL_PATH" "$REMOTE_NAME:$REMOTE_PATH" --progress

echo "✅ Transfert terminé."
