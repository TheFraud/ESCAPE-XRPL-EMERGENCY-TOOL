#!/bin/bash
# Script pour créer l'archive téléchargeable
cd /home/rootsider/escape/dist || exit
zip -r escape_wallet.zip .
echo "Archive escape_wallet.zip créée avec succès dans $(pwd)"
