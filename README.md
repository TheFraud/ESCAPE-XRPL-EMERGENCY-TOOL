# ESCAPE-XRPL-EMERGENCY-TOOL
XRPL EMERGENCY TOOL ACCESS
### ESCAPE – XRP Ledger Emergency Access Tool / 
/ 
ESCAPE est un outil complet et sécurisé d’accès aux portefeuilles XRP Ledger. Il combine des connexions réseaux hybrides (WebSocket et JSON‑RPC), des fonctions cryptographiques robustes et une interface graphique interactive réalisée en Tkinter. Ce document présente en détail l’architecture, les fonctionnalités et les instructions d’utilisation de l’application.

Description / Description

FR :
L’application ESCAPE permet de gérer des portefeuilles XRP avec des fonctionnalités telles que la création, la récupération (via seed), l’envoi de transactions et la consultation d’historique. Le tout est sécurisé par des méthodes de chiffrement (PBKDF2 + Fernet) pour protéger les seeds et les données sensibles. En plus de son module de gestion réseau, l’outil offre un environnement graphique animé qui illustre l’activité du réseau et affiche des indicateurs en temps réel (connexion, latence, état du ledger, etc.). L’architecture asynchrone permet d’assurer une communication fluide et de basculer entre plusieurs endpoints en cas de défaillance.

 EN :
ESCAPE is a comprehensive and secure, simple  XRP Ledger wallet access tool. It enables wallet creation, seed recovery, transaction sending, and history checking, all secured with robust encryption methods (PBKDF2 and Fernet) to protect seeds and sensitive data. In addition to its network management module, the tool features an interactive and animated graphical interface that visually represents network activity and displays real-time metrics (connection status, latency, ledger state, etc.). An asynchronous architecture ensures smooth communication and automatic endpoint switching in case of network issues.
Fonctionnalités / Key Features
FR :

    Portefeuille XRP sécurisé : Création de nouvelles portes (wallets) et récupération via seed.
    Réseau hybride : Communication simultanée via WebSocket et JSON‑RPC permettant une redondance (fallback) et une reconnexion automatique aux endpoints XRPL.
    Cryptographie : Utilisation de PBKDF2 et Fernet pour chiffrer et déchiffrer les données sensibles (e.g. seed).
    Monitoring & History : Consultation en temps réel de l’état du réseau, de la latence des requêtes, des informations sur le ledger et de l’historique des transactions.
    Interface graphique animée : Animation réseau (NetworkAnimation) et horloge dynamique, avec un environnement inspiré de l’univers « Matrix ».
    Gestion multi-threadée et asynchrone : Support d’asyncio et utilisation de threads pour garantir une interface réactive tout en réalisant des opérations réseau en arrière-plan.

EN :

    Secure XRP Wallet: Create new “doors” (wallets) and recover them using a seed.
    Hybrid Networking: Simultaneous WebSocket and JSON‑RPC communication allowing redundancy (fallback) and automatic reconnection to XRPL endpoints.
    Cryptography: Use of PBKDF2 and Fernet for encrypting and decrypting sensitive data (e.g. wallet seed).
    Monitoring & History: Real-time monitoring of network status, request latency, ledger information, and transaction history.
    Animated Graphical Interface: Network animations (NetworkAnimation) and a dynamic clock within a “Matrix‑inspired” environment.
    Asynchronous & Multi-threaded Architecture: Utilizes asyncio and threads to maintain a responsive GUI while performing background network operations.

Technologies et Dépendances / Technologies and Dependencies
FR :

    Langage : Python 3.7+
    Bibliothèques XRP :
    • xrpl-py (gestion du wallet et transactions XRP)
    Modules Cryptographiques :
    • cryptography (PBKDF2, Fernet)
    Communication Réseau :
    • websockets (connexion WebSocket)
    Interface Graphique :
    • tkinter, PIL (Pillow)
    Autres Modules :
    • asyncio, threading, logging, datetime, etc.

EN :

    Language: Python 3.7+
    XRP Libraries:
    • xrpl-py (wallet management and XRP transactions)
    Cryptography Modules:
    • cryptography (PBKDF2, Fernet)
    Network Communication:
    • websockets (WebSocket connection)
    Graphical Interface:
    • tkinter, PIL (Pillow)
    Other Modules:
    • asyncio, threading, logging, datetime, etc.




Structure du Code / Code Structure
L’architecture est organisée en plusieurs modules et classes :
1. Cryptographie et Sécurité

    Fonctions :
    • generate_key(seed, salt) : Dérive une clé sécurisée à partir d’un seed et d’un sel.
    • encrypt_data(data, seed) : Chiffre les données sensibles en préfixant le résultat avec le sel.
    • decrypt_data(encrypted_data, seed) : Récupère et déchiffre les données.

2. Communication avec le Réseau XRP Ledger

    Classes :
    • HybridXRPLClient : Gère la connexion hybride via WebSocket et HTTP JSON‑RPC.
    – Implémente un système de gestion asynchrone (via asyncio) pour recevoir des mises à jour et envoyer des requêtes synchrones en mode fallback.
    – Intègre une logique de reconnexion automatique et de gestion des messages en arrière-plan.
    Fonction :
    • reliable_submission(signed_tx, json_client, timeout) : Soumet une transaction signée et attend sa validation sur le réseau grâce à une requête répétée vérifiant le statut.

3. Gestion du Portefeuille et des Transactions XRP

    Classes :
    • MatrixEscape : Le gestionnaire central qui orchestre :
        Le passage entre le mode offline et online
        La création d’un nouveau portefeuille (méthode create_escape)
        La récupération d’informations (balance, ledger, état du serveur)
        L’envoi de transactions XRP via la méthode send_xrp • DoorWallet : Fournit des méthodes pour créer ou récupérer un portefeuille via un seed.

4. Interface Graphique et Animations

    Classes UI/Animations : • EscapeGUI : Gère l’interface Tkinter permettant de naviguer entre les différentes actions (balance, envoi de XRP, historique, gestion du wallet). • NetworkAnimation : Anime un réseau de nœuds mobiles pour visualiser l’activité. • GlobeAnimation : Affiche une animation circulaire avec des marqueurs numériques pour renforcer l’aspect dynamique et immersif de l’application.

5. Exécution Asynchrone et Multi-threadée

    L’utilisation d’asyncio et de threads permet de gérer simultanément la communication réseau en arrière-plan (via HybridXRPLClient et la surveillance continue dans MatrixEscape) et l’actualisation de l’interface graphique.

Points Clés / Key Points

    Sécurité :
    L’application accorde une grande importance à la sécurisation des données sensibles grâce aux algorithmes cryptographiques robustes et à une gestion prudente des clés privées.
    Redondance et Fiabilité :
    Le système hybride permettant de passer du protocole WebSocket au HTTP JSON‑RPC garantit une communication fiable et une tolérance aux pannes grâce à la reconnexion automatique et à la gestion dynamique des endpoints XRPL.
    Interface Utilisateur :
    L’interface graphique sous Tkinter, agrémentée d’animations interactives et d’un monitoring en temps réel, offre une expérience immersive tout en présentant clairement l’état du réseau et des transactions.


Contributing

FR :
Les contributions sont les bienvenues. Si vous souhaitez améliorer le code, corriger des bugs, ou ajouter de nouvelles fonctionnalités, merci de soumettre un « Pull Request » sur le dépôt.

 EN :
Contributions are welcome. If you wish to enhance the code, fix bugs, or add new features, please submit a pull request on the repository.


SUPPORT /No need support , Now i took my meds !!!! / Only contributing We never no what come next !?

preview
https://www.youtube.com/@escape_404
