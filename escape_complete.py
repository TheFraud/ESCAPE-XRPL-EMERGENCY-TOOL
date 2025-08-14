#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESCAPE - XRP Ledger Emergency Access Tool
MAINNET ONLY VERSION - 100% REAL XRP

Améliorations incluses :
- Installation conditionnelle des dépendances (installées uniquement si nécessaire)
- Envoi de transactions avec autofill (gestion dynamique du Sequence, Fee, LastLedgerSequence)
- Validation d’adresse XRPL (adresse classique)
- Gestion de la réserve : calcul du solde dépensable versus total
- Destination Tag (optionnel) pour les paiements
- Sauvegarde et chargement du wallet chiffré (Fernet + PBKDF2, avec header "ESCAPE1")
- Affichage du prix XRP en USD via CoinGecko (avec cache)
- Animation réseau optimisée adaptée à la taille du canvas
- Interface graphique complète avec Tkinter et gestion de wallet (création, récupération, ouverture, envoi, historique)
- Bouton "Update Wallet" pour rafraîchir manuellement les infos du wallet
- Messages d’erreurs détaillés et robustesse globale accrue
"""

import sys
import subprocess
import os
import base64
import json
import random
import math
import time
import threading
import logging
from datetime import datetime
from itertools import cycle
from typing import Dict, List, Any, Optional, Tuple
from decimal import Decimal, InvalidOperation, ROUND_DOWN

# ==============================================================================
# CONFIGURATION DU LOG
# ==============================================================================
log_file = os.path.expanduser('~/.escape_mainnet.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logger = logging.getLogger("ESCAPE")

# ==============================================================================
# GESTION DES DÉPENDANCES
# ==============================================================================
def ensure_dependencies():
    """Installe les dépendances requises si non déjà installées."""
    dependencies = [
        'xrpl-py>=2.0.0',
        'cryptography>=36.0.0',
        'websockets>=10.0',
        'requests>=2.27.0'
    ]
    try:
        import xrpl  # noqa
        import cryptography  # noqa
        import requests  # noqa
    except Exception:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + dependencies)
            print("✅ Dépendances installées avec succès.")
        except Exception as e:
            print(f"❌ Erreur lors de l'installation des dépendances: {e}")
            sys.exit(1)

ensure_dependencies()

# ==============================================================================
# IMPORTS
# ==============================================================================
try:
    # XRPL
    from xrpl.clients import JsonRpcClient
    from xrpl.wallet import Wallet
    from xrpl.models.requests import AccountInfo, ServerInfo, AccountTx
    from xrpl.models.transactions import Payment
    from xrpl.utils import xrp_to_drops, drops_to_xrp
    from xrpl.transaction import submit_and_wait, autofill
    from xrpl.core.addresscodec import is_valid_classic_address

    # Cryptographie
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.backends import default_backend
    from cryptography.fernet import Fernet

    # Interface graphique (Tkinter)
    import tkinter as tk
    from tkinter import messagebox, simpledialog, ttk, scrolledtext

    # Réseaux
    import requests
    WEBSOCKETS_AVAILABLE = True
    try:
        import websockets  # noqa
    except ImportError:
        WEBSOCKETS_AVAILABLE = False
        logger.warning("WebSockets non disponibles, utilisation de JSON-RPC uniquement")
except ImportError as e:
    print(f"❌ Erreur d'importation: {e}")
    sys.exit(1)

# ==============================================================================
# CONSTANTES
# ==============================================================================
APP_VERSION = "1.2.0"
APP_NAME = "ESCAPE"

# XRPL endpoints mainnet
MAINNET_ENDPOINTS = [
    "https://xrplcluster.com/",
    "https://s1.ripple.com:51234/",
    "https://s2.ripple.com:51234/"
]
MAINNET_WS_ENDPOINTS = [
    "wss://xrplcluster.com/",
    "wss://s1.ripple.com/",
    "wss://s2.ripple.com/"
]

# Chemin de sauvegarde par défaut du wallet chiffré
DEFAULT_KEYSTORE_PATH = os.path.expanduser("~/.escape_mainnet.wallet")

# ==============================================================================
# FONCTIONS DE SÉCURITÉ / CHIFFREMENT
# ==============================================================================
def generate_key(password: str, salt: bytes) -> bytes:
    """Génère une clé d'encryptage à partir d'un mot de passe et d'un sel."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    return base64.urlsafe_b64encode(kdf.derive(password.encode()))

def encrypt_data(data: str, password: str) -> bytes:
    """Chiffre les données avec le mot de passe; on ajoute l'en-tête 'ESCAPE1'."""
    salt = os.urandom(16)
    key = generate_key(password, salt)
    fernet = Fernet(key)
    encrypted = fernet.encrypt(data.encode())
    return b"ESCAPE1" + salt + encrypted

def decrypt_data(encrypted_data: bytes, password: str) -> str:
    """Déchiffre les données avec le mot de passe."""
    if not encrypted_data.startswith(b"ESCAPE1"):
        raise ValueError("Ce fichier n'est pas un keystore ESCAPE valide.")
    encrypted_data = encrypted_data[len(b"ESCAPE1"):]
    salt = encrypted_data[:16]
    actual_encrypted = encrypted_data[16:]
    key = generate_key(password, salt)
    fernet = Fernet(key)
    decrypted = fernet.decrypt(actual_encrypted)
    return decrypted.decode()

class SecureStore:
    """Sauvegarde et chargement chiffré du wallet en local."""
    def __init__(self, path: Optional[str] = None):
        self.path = path or DEFAULT_KEYSTORE_PATH

    def save(self, wallet: dict, password: str) -> bool:
        if not wallet or 'seed' not in wallet or 'address' not in wallet:
            raise ValueError("Wallet invalide pour la sauvegarde.")
        payload = {
            "v": 1,
            "created_at": datetime.now().isoformat(),
            "address": wallet['address'],
            "seed": wallet['seed'],
        }
        data = json.dumps(payload)
        enc = encrypt_data(data, password)
        with open(self.path, "wb") as f:
            f.write(enc)
        return True

    def load(self, password: str) -> dict:
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Aucun keystore trouvé: {self.path}")
        with open(self.path, "rb") as f:
            enc = f.read()
        data = decrypt_data(enc, password)
        payload = json.loads(data)
        if "seed" not in payload or "address" not in payload:
            raise ValueError("Keystore corrompu ou incomplet.")
        return {
            "address": payload["address"],
            "seed": payload["seed"],
            "created_at": payload.get("created_at", "Inconnu")
        }

# ==============================================================================
# CLIENT XRPL MAINNET AVEC FAILOVER AUTOMATIQUE
# ==============================================================================
class MainnetClient:
    """Client XRPL pour le mainnet avec basculement automatique en cas d'erreur de connexion."""
    def __init__(self):
        self.http_endpoints = MAINNET_ENDPOINTS[:]
        self.endpoint_cycle = cycle(self.http_endpoints)
        self.current_endpoint = next(self.endpoint_cycle)
        self.client = JsonRpcClient(self.current_endpoint)

        # Suivi d'état
        self.connection_status = False
        self.connection_info = {"server_state": "UNKNOWN", "build_version": "UNKNOWN"}
        self.connection_latency = 0
        self.ledger_info = {"ledger_current_index": "UNKNOWN"}
        self.server_load = "UNKNOWN"
        self.connected_nodes = "UNKNOWN"
        self.failed_attempts = 0

        # Cache du prix XRP en USD
        self._xrp_price_usd = None
        self._last_price_fetch = 0

    def switch_endpoint(self):
        """Bascule vers l'endpoint suivant."""
        self.current_endpoint = next(self.endpoint_cycle)
        logger.info(f"Switching to mainnet endpoint: {self.current_endpoint}")
        self.client = JsonRpcClient(self.current_endpoint)
        self.failed_attempts = 0

    def check_connection(self) -> Tuple[bool, Dict]:
        """Vérifie la connexion au mainnet XRPL."""
        try:
            start_time = time.time()
            response = self.client.request(ServerInfo())
            if not response.is_successful():
                raise ConnectionError("Le ServerInfo a échoué")
            latency = round((time.time() - start_time) * 1000)
            self.connection_latency = latency

            result = response.result
            self.connection_info = result.get("info", {})
            server_state = self.connection_info.get("server_state", "UNKNOWN")
            self.server_load = self.connection_info.get("load_factor", "UNKNOWN")
            self.connection_status = True

            # Récupération d'informations complémentaires
            self._get_ledger_info()
            self._get_network_nodes()

            self.failed_attempts = 0
            return True, {
                "state": server_state,
                "version": self.connection_info.get("build_version", "UNKNOWN"),
                "latency": latency,
                "endpoint": self.current_endpoint
            }
        except Exception as e:
            logger.error(f"Connection error: {str(e)}")
            self.connection_status = False
            self.failed_attempts += 1
            if self.failed_attempts >= 3:
                self.switch_endpoint()
            return False, {"error": str(e), "endpoint": self.current_endpoint}

    def _get_ledger_info(self):
        """Récupère l'index du ledger courant."""
        try:
            response = self.client.request({"command": "ledger_current"})
            if response.is_successful():
                self.ledger_info = response.result
        except Exception as e:
            logger.error(f"Error fetching ledger info: {e}")

    def _get_network_nodes(self):
        """Récupère le nombre de peers connectés."""
        try:
            response = self.client.request({"command": "peers"})
            if response.is_successful() and "peers" in response.result:
                self.connected_nodes = len(response.result["peers"])
        except Exception as e:
            logger.error(f"Error fetching peers: {e}")

    def get_reserve_params(self) -> Tuple[Decimal, Decimal]:
        """Retourne (reserve_base_xrp, reserve_inc_xrp) en Decimal."""
        try:
            response = self.client.request(ServerInfo())
            if response.is_successful():
                info = response.result.get("info", {})
                v = info.get("validated_ledger", {})
                base = Decimal(str(v.get("reserve_base_xrp", "10")))
                inc = Decimal(str(v.get("reserve_inc_xrp", "2")))
                return base, inc
        except Exception as e:
            logger.warning(f"Impossible de récupérer les paramètres de réserve, utilisation des valeurs par défaut. {e}")
        return Decimal("10"), Decimal("2")

    def get_account_info(self, address: str) -> Optional[dict]:
        try:
            response = self.client.request(AccountInfo(
                account=address,
                ledger_index="validated",
                strict=True
            ))
            if response.is_successful():
                return response.result.get("account_data")
        except Exception as e:
            logger.error(f"Error in get_account_info: {e}")
        return None

    def get_account_balance(self, address: str) -> Decimal:
        """Retourne le solde XRP d'un compte en Decimal."""
        try:
            response = self.client.request(AccountInfo(
                account=address,
                ledger_index="validated",
                strict=True
            ))
            if not response.is_successful():
                return Decimal("0")
            balance_drops = response.result["account_data"]["Balance"]
            return Decimal(str(drops_to_xrp(balance_drops)))
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return Decimal("0")

    def calculate_spendable_balance(self, address: str) -> Tuple[Decimal, Decimal, int]:
        """
        Retourne (balance_total, balance_depensable, owner_count).
        balance_depensable = balance_total - (reserve_base + owner_count * reserve_inc)
        """
        try:
            account_info = self.get_account_info(address)
            if not account_info:
                return Decimal("0"), Decimal("0"), 0
            balance_xrp = Decimal(str(drops_to_xrp(account_info["Balance"])))
            owner_count = int(account_info.get("OwnerCount", 0))
            base, inc = self.get_reserve_params()
            reserve = base + (inc * Decimal(owner_count))
            spendable = balance_xrp - reserve
            if spendable < Decimal("0"):
                spendable = Decimal("0")
            return balance_xrp, spendable, owner_count
        except Exception as e:
            logger.error(f"Error calculating spendable balance: {e}")
            return Decimal("0"), Decimal("0"), 0

    def get_xrp_price_usd(self) -> Optional[float]:
        """Récupère le prix du XRP en USD depuis CoinGecko (mise en cache pendant 60 sec)."""
        try:
            now = time.time()
            if self._xrp_price_usd is not None and (now - self._last_price_fetch) < 60:
                return self._xrp_price_usd
            url = "https://api.coingecko.com/api/v3/simple/price"
            params = {"ids": "ripple", "vs_currencies": "usd"}
            resp = requests.get(url, params=params, timeout=5)
            if resp.ok:
                data = resp.json()
                price = float(data.get("ripple", {}).get("usd", 0))
                if price > 0:
                    self._xrp_price_usd = price
                    self._last_price_fetch = now
                    return price
        except Exception as e:
            logger.warning(f"Price fetch failed: {e}")
        return None

    def send_xrp(self, sender_wallet: dict, destination: str, amount_xrp: Decimal,
                 destination_tag: Optional[int] = None) -> dict:
        """Effectue un paiement XRP réel avec validation et autofill."""
        try:
            if not is_valid_classic_address(destination):
                return {"success": False, "error": "Adresse XRPL invalide."}
            if amount_xrp <= Decimal("0"):
                return {"success": False, "error": "Montant invalide (doit être > 0)."}
            total, spendable, _ = self.calculate_spendable_balance(sender_wallet['address'])
            if amount_xrp > spendable:
                return {"success": False, "error": f"Fonds insuffisants. Dépensable: {spendable} XRP, Total: {total} XRP"}
            drops_amount = xrp_to_drops(str(amount_xrp))
            tx_kwargs = {
                "account": sender_wallet['address'],
                "destination": destination,
                "amount": drops_amount
            }
            if destination_tag is not None:
                if not isinstance(destination_tag, int) or not (0 <= destination_tag < 2**32):
                    return {"success": False, "error": "Destination Tag invalide."}
                tx_kwargs["destination_tag"] = destination_tag
            payment = Payment(**tx_kwargs)
            # Utilisation d'autofill pour compléter Sequence, Fee, etc.
            payment = autofill(self.client, payment)
            wallet_obj = Wallet(seed=sender_wallet['seed'])
            logger.info(f"Envoi de {amount_xrp} XRP à {destination} (tag={destination_tag})")
            response = submit_and_wait(payment, self.client, wallet_obj)
            if response.is_successful():
                result = response.result
                tx_hash = result.get("hash") or result.get("tx_json", {}).get("hash")
                ledger_idx = result.get("ledger_index") or result.get("validated_ledger_index")
                logger.info(f"Transaction réussie: {tx_hash}")
                return {
                    "success": True,
                    "hash": tx_hash,
                    "ledger": ledger_idx,
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            else:
                error = response.result.get("engine_result_message", "Erreur inconnue")
                logger.error(f"Echec de la transaction: {error}")
                return {"success": False, "error": error}
        except Exception as e:
            logger.error(f"Error sending XRP: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}

    def get_transaction_history(self, address: str, limit: int = 10) -> List[Dict]:
        """Retourne l'historique des transactions d'un compte."""
        try:
            response = self.client.request(AccountTx(
                account=address,
                ledger_index_min=-1,
                ledger_index_max=-1,
                limit=limit,
                binary=False
            ))
            if not response.is_successful():
                return []
            transactions = []
            for tx_info in response.result.get("transactions", []):
                tx = tx_info.get("tx", {})
                meta = tx_info.get("meta", {})
                date_str = "Unknown"
                if "date" in tx:
                    try:
                        timestamp = tx["date"] + 946684800
                        date_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
                    except Exception:
                        pass
                amount = "0"
                if "Amount" in tx:
                    try:
                        if isinstance(tx["Amount"], str):
                            amount = f"{drops_to_xrp(tx['Amount'])} XRP"
                        else:
                            amount = f"{tx['Amount']['value']} {tx['Amount']['currency']}"
                    except Exception:
                        amount = str(tx.get("Amount", "0"))
                transactions.append({
                    "hash": tx.get("hash", "Unknown"),
                    "date": date_str,
                    "type": tx.get("TransactionType", "Unknown"),
                    "amount": amount,
                    "fee": str(drops_to_xrp(tx.get("Fee", "0"))) + " XRP",
                    "result": meta.get("TransactionResult", "Unknown"),
                    "validated": tx_info.get("validated", False)
                })
            return transactions
        except Exception as e:
            logger.error(f"Error getting transaction history: {e}")
            return []

    def get_network_load(self) -> str:
        """Retourne le facteur de charge du réseau."""
        return str(self.server_load)

    def get_connected_nodes(self) -> str:
        """Retourne le nombre de nœuds connectés."""
        return str(self.connected_nodes)

# ==============================================================================
# ANIMATION RÉSEAU OPTIMISÉE
# ==============================================================================
class NetworkAnimation:
    """Visualisation en temps réel du réseau XRPL avec animation fluide."""
    def __init__(self, canvas, width=None, height=None, xrpl_client=None):
        self.canvas = canvas
        self.width = width if width and width > 100 else 800
        self.height = height if height and height > 100 else 600
        self.xrpl_client = xrpl_client

        self.last_time = time.time()
        self.node_ids = {}
        self.line_ids = {}
        self.text_ids = {}
        self.pulse_counter = 0

        self.nodes = []
        self.node_count = 18
        self.connection_distance = 150
        self.active_nodes = 0

        self._create_nodes()
        self._create_ui_elements()
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.update()

    def _on_canvas_resize(self, event):
        if event.width > 50 and event.height > 50:
            self.width = event.width
            self.height = event.height
            for node in self.nodes:
                if node['x'] < 10 or node['x'] > self.width - 10:
                    node['x'] = random.randint(20, self.width - 20)
                if node['y'] < 10 or node['y'] > self.height - 10:
                    node['y'] = random.randint(20, self.height - 20)

    def _create_nodes(self):
        for i in range(self.node_count):
            x = random.randint(20, max(21, self.width - 20))
            y = random.randint(20, max(21, self.height - 20))
            node = {
                'id': i,
                'x': x,
                'y': y,
                'dx': random.uniform(-0.6, 0.6),
                'dy': random.uniform(-0.6, 0.6),
                'size': random.randint(2, 4),
                'active': random.choice([True, False]),
                'canvas_id': None
            }
            self.nodes.append(node)
            color = "#00FF00" if node['active'] else "#003300"
            node_id = self.canvas.create_oval(
                node['x']-node['size'], node['y']-node['size'],
                node['x']+node['size'], node['y']+node['size'],
                fill=color, outline=color
            )
            node['canvas_id'] = node_id
            self.node_ids[i] = node_id

    def _create_ui_elements(self):
        self.status_text = self.canvas.create_text(
            10, 10,
            text="INITIALIZING XRPL CONNECTION...",
            fill="#00FF00", anchor="nw", font=("Courier", 10)
        )
        self.connection_indicator = self.canvas.create_oval(
            10, 35, 20, 45,
            fill="#333333", outline="#00FF00"
        )
        y_pos = 65
        status_items = ["status", "latency", "server", "ledger", "load", "nodes"]
        for item in status_items:
            self.text_ids[item] = self.canvas.create_text(
                10, y_pos,
                text=f"{item.upper()}: ...",
                fill="#00FF00", anchor="nw", font=("Courier", 9)
            )
            y_pos += 18

    def update(self):
        current_time = time.time()
        dt = min(current_time - self.last_time, 0.05)
        self.last_time = current_time
        self.pulse_counter = (self.pulse_counter + dt*2) % (2*math.pi)

        connection_active = False
        if self.xrpl_client:
            connection_active = self.xrpl_client.connection_status
            if connection_active:
                self._update_status_texts()
            status_text = "CONNECTED" if connection_active else "DISCONNECTED"
            self.canvas.itemconfig(
                self.text_ids.get('status', 0),
                text=f"STATUS: {status_text}",
                fill="#00FF00" if connection_active else "#FF0000"
            )
        self._update_connection_indicator(connection_active)
        self._update_nodes(dt, connection_active)
        active_connections = self._update_connections()
        self.canvas.itemconfig(
            self.status_text,
            text=f"XRPL {'ONLINE' if connection_active else 'OFFLINE'} | ACTIVE: {self.active_nodes}/{self.node_count} | CONNECTIONS: {active_connections}"
        )
        self.canvas.after(16, self.update)

    def _update_status_texts(self):
        if not self.xrpl_client:
            return
        ledger_index = self.xrpl_client.ledger_info.get('ledger_current_index', 'UNKNOWN')
        self.canvas.itemconfig(
            self.text_ids.get('ledger', 0),
            text=f"LEDGER: {ledger_index}"
        )
        latency = self.xrpl_client.connection_latency
        latency_color = "#FF0000" if latency > 500 else ("#FFFF00" if latency > 200 else "#00FF00")
        self.canvas.itemconfig(
            self.text_ids.get('latency', 0),
            text=f"LATENCY: {latency} ms",
            fill=latency_color
        )
        server_state = self.xrpl_client.connection_info.get('server_state', 'UNKNOWN')
        self.canvas.itemconfig(
            self.text_ids.get('server', 0),
            text=f"SERVER: {server_state}"
        )
        load = self.xrpl_client.get_network_load()
        nodes = self.xrpl_client.get_connected_nodes()
        self.canvas.itemconfig(
            self.text_ids.get('load', 0),
            text=f"LOAD: {load}"
        )
        self.canvas.itemconfig(
            self.text_ids.get('nodes', 0),
            text=f"NODES: {nodes}"
        )

    def _update_connection_indicator(self, connection_active):
        pulse_alpha = (math.sin(self.pulse_counter) + 1) / 2
        if connection_active:
            r = 0
            g = int(128 + 127 * pulse_alpha)
            b = 0
        else:
            r = int(128 + 127 * pulse_alpha)
            g = 0
            b = 0
        color = f"#{r:02x}{g:02x}{b:02x}"
        self.canvas.itemconfig(self.connection_indicator, fill=color)

    def _update_nodes(self, dt, connection_active):
        self.active_nodes = 0
        speed_factor = 60 * dt
        for node in self.nodes:
            node['x'] += node['dx'] * speed_factor
            node['y'] += node['dy'] * speed_factor
            if node['x'] < node['size'] or node['x'] > self.width - node['size']:
                node['dx'] *= -1
                node['dx'] += random.uniform(-0.05, 0.05)
            if node['y'] < node['size'] or node['y'] > self.height - node['size']:
                node['dy'] *= -1
                node['dy'] += random.uniform(-0.05, 0.05)
            speed = math.sqrt(node['dx']**2 + node['dy']**2)
            if speed > 1.5:
                node['dx'] = node['dx'] * 1.5 / speed
                node['dy'] = node['dy'] * 1.5 / speed
            if connection_active:
                if random.random() < 0.005:
                    node['active'] = not node['active']
            elif random.random() < 0.002:
                node['active'] = random.random() < 0.3
            color = "#00FF00" if node['active'] else "#003300"
            if node['active']:
                self.active_nodes += 1
            self.canvas.itemconfig(node['canvas_id'], fill=color, outline=color)
            self.canvas.coords(
                node['canvas_id'],
                node['x']-node['size'], node['y']-node['size'],
                node['x']+node['size'], node['y']+node['size']
            )

    def _update_connections(self):
        for line_id in self.line_ids.values():
            self.canvas.itemconfig(line_id, state='hidden')
        active_connections = 0
        for i, node1 in enumerate(self.nodes):
            for j in range(i+1, len(self.nodes)):
                node2 = self.nodes[j]
                dx = node1['x'] - node2['x']
                dy = node1['y'] - node2['y']
                if dx*dx + dy*dy < self.connection_distance * self.connection_distance:
                    conn_key = f"{i}_{j}"
                    active_conn = node1['active'] and node2['active']
                    if active_conn:
                        active_connections += 1
                    line_color = "#00FF00" if active_conn else "#003300"
                    line_width = 1.0 if active_conn else 0.5
                    if conn_key in self.line_ids:
                        line_id = self.line_ids[conn_key]
                        self.canvas.itemconfig(
                            line_id,
                            fill=line_color,
                            width=line_width,
                            state='normal'
                        )
                        self.canvas.coords(
                            line_id,
                            node1['x'], node1['y'], node2['x'], node2['y']
                        )
                    else:
                        line_id = self.canvas.create_line(
                            node1['x'], node1['y'], node2['x'], node2['y'],
                            fill=line_color, width=line_width
                        )
                        self.line_ids[conn_key] = line_id
        return active_connections

# ==============================================================================
# GESTION DU WALLET (Création, Récupération, Paiement, etc.)
# ==============================================================================
class WalletManager:
    """Gestion des opérations de wallet XRPL."""
    def __init__(self, xrpl_client):
        self.xrpl_client = xrpl_client
        self.keystore = SecureStore()

    def create_wallet(self) -> dict:
        wallet = Wallet.create()
        return {
            'address': wallet.classic_address,
            'seed': wallet.seed,
            'public_key': wallet.public_key,
            'private_key': wallet.private_key,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    def recover_wallet(self, seed: str) -> dict:
        try:
            wallet = Wallet(seed=seed)
            return {
                'address': wallet.classic_address,
                'seed': seed,
                'created_at': 'Recovered wallet'
            }
        except Exception as e:
            raise ValueError(f"Seed invalide: {e}")

    def get_balance(self, address: str) -> Decimal:
        return self.xrpl_client.get_account_balance(address)

    def send_payment(self, wallet: dict, destination: str, amount: Decimal, destination_tag: Optional[int] = None) -> dict:
        return self.xrpl_client.send_xrp(wallet, destination, amount, destination_tag=destination_tag)

    def get_transactions(self, address: str, limit: int = 20) -> List[Dict]:
        return self.xrpl_client.get_transaction_history(address, limit)

    def save_encrypted(self, wallet: dict, password: str) -> bool:
        return self.keystore.save(wallet, password)

    def load_encrypted(self, password: str) -> dict:
        return self.keystore.load(password)

# ==============================================================================
# APPLICATION PRINCIPALE (INTERFACE TKINTER)
# ==============================================================================
class EscapeWallet:
    """Application principale du portefeuille ESCAPE."""
    def __init__(self, root):
        self.root = root
        self.root.title(f"{APP_NAME} Wallet v{APP_VERSION} - MAINNET (REAL XRP)")
        self.root.geometry("1200x800")
        self.root.configure(bg="black")

        self.xrpl_client = MainnetClient()
        self.wallet_manager = WalletManager(self.xrpl_client)
        self.current_wallet = None

        self.create_interface()
        self.start_updates()

    def start_updates(self):
        self.update_connection()
        self.root.after(1000, lambda: self.check_client_connection())

    def update_connection(self):
        try:
            self.root.after(2000, self.update_connection)
        except Exception as e:
            logger.error(f"Error in update_connection: {e}")
            self.root.after(2000, self.update_connection)

    def check_client_connection(self):
        try:
            self.xrpl_client.check_connection()
            self.root.after(10000, self.check_client_connection)
        except Exception as e:
            logger.error(f"Error checking client connection: {e}")
            self.root.after(10000, self.check_client_connection)

    def create_interface(self):
        main_frame = tk.Frame(self.root, bg="black")
        main_frame.pack(expand=True, fill="both")

        self.left_sidebar = self.create_sidebar(main_frame)
        self.content_area = tk.Frame(main_frame, bg="black")
        self.content_area.pack(side="left", expand=True, fill="both")

        self.animation_canvas = tk.Canvas(self.content_area, bg="black", highlightthickness=0)
        self.animation_canvas.pack(expand=True, fill="both")

        self.network_animation = NetworkAnimation(
            self.animation_canvas,
            800,
            600,
            xrpl_client=self.xrpl_client
        )

    def create_sidebar(self, parent):
        sidebar = tk.Frame(parent, bg="black", width=300)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        tk.Label(
            sidebar,
            text=APP_NAME,
            font=("Courier", 24, "bold"),
            fg="#00FF00",
            bg="black"
        ).pack(pady=(20, 5))

        tk.Label(
            sidebar,
            text="MAINNET - REAL XRP",
            font=("Courier", 12, "bold"),
            fg="#FF0000",
            bg="black"
        ).pack(pady=(0, 20))

        warning_frame = tk.Frame(sidebar, bg="#330000", bd=2, relief="raised")
        warning_frame.pack(fill="x", pady=10, padx=10)
        tk.Label(
            warning_frame,
            text="⚠️ WARNING: REAL MONEY ⚠️",
            font=("Courier", 12, "bold"),
            fg="#FFFFFF",
            bg="#330000"
        ).pack(pady=5)
        tk.Label(
            warning_frame,
            text="This application uses REAL XRP\nTransactions are IRREVERSIBLE",
            font=("Courier", 10),
            fg="#FFFFFF",
            bg="#330000"
        ).pack(pady=5)

        status_frame = tk.Frame(sidebar, bg="black")
        status_frame.pack(fill="x", pady=15, padx=10)
        self.wallet_status = tk.Label(
            status_frame,
            text="No Wallet Loaded",
            font=("Courier", 11),
            fg="#FFFF00",
            bg="black"
        )
        self.wallet_status.pack(fill="x")

        menu_frame = tk.Frame(sidebar, bg="black")
        menu_frame.pack(fill="both", expand=True, padx=10, pady=10)

        wallet_group = tk.LabelFrame(
            menu_frame,
            text="Wallet",
            font=("Courier", 10, "bold"),
            fg="#00FF00",
            bg="black",
            bd=1
        )
        wallet_group.pack(fill="x", pady=10)

        def add_button(parent, text, command, color="#00FF00"):
            btn = tk.Button(
                parent,
                text=text,
                font=("Courier", 11),
                fg=color,
                bg="black",
                activebackground="#001100",
                activeforeground=color,
                relief="flat",
                borderwidth=1,
                command=command
            )
            btn.pack(fill="x", pady=2, padx=5)
            return btn

        add_button(wallet_group, "🆕 Create Wallet", self.create_wallet_ui)
        add_button(wallet_group, "🔓 Open Wallet (Seed)", self.open_wallet_ui)
        add_button(wallet_group, "💾 Save Encrypted Wallet", self.save_encrypted_wallet_ui)
        add_button(wallet_group, "📂 Load Encrypted Wallet", self.load_encrypted_wallet_ui)
        add_button(wallet_group, "💼 Wallet Info", self.show_wallet_info)
        add_button(wallet_group, "🔄 Update Wallet", self.update_wallet_info)
        add_button(wallet_group, "🔒 Disconnect", self.disconnect_wallet, color="#FF0000")

        tx_group = tk.LabelFrame(
            menu_frame,
            text="Transactions",
            font=("Courier", 10, "bold"),
            fg="#00FF00",
            bg="black",
            bd=1
        )
        tx_group.pack(fill="x", pady=10)

        add_button(tx_group, "💰 View Balance", self.view_balance)
        add_button(tx_group, "💸 Send XRP", self.send_xrp_ui)
        add_button(tx_group, "📜 Transaction History", self.show_history)

        return sidebar

    def update_wallet_status(self):
        if self.current_wallet:
            addr = self.current_wallet['address']
            short_addr = addr[:8] + "..." + addr[-8:]
            self.wallet_status.config(text=f"Wallet: {short_addr}", fg="#00FF00")
        else:
            self.wallet_status.config(text="No Wallet Loaded", fg="#FFFF00")

    def create_wallet_ui(self):
        if not messagebox.askyesno(
            "REAL MAINNET WALLET",
            "⚠️ WARNING ⚠️\n\nVous allez créer un wallet sur le XRPL mainnet.\n"
            "Ce wallet aura 0 XRP jusqu'à ce que vous le financiez.\n\n"
            "Voulez-vous continuer ?"
        ):
            return
        try:
            wallet_data = self.wallet_manager.create_wallet()
            self.current_wallet = wallet_data
            self.update_wallet_status()
            self.show_seed_backup(wallet_data)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create wallet: {str(e)}")

    def show_seed_backup(self, wallet_data):
        dialog = tk.Toplevel(self.root)
        dialog.title("CRITICAL: Save Your Seed")
        dialog.configure(bg="black")
        dialog.geometry("600x420")
        tk.Label(
            dialog,
            text="⚠️ IMPORTANT: SAVE THIS SEED! ⚠️",
            font=("Courier", 16, "bold"),
            fg="#FF0000",
            bg="black"
        ).pack(pady=10)
        tk.Label(
            dialog,
            text="This is your only recovery method!",
            font=("Courier", 12),
            fg="#FFFF00",
            bg="black"
        ).pack(pady=5)
        tk.Label(
            dialog,
            text="Address:",
            font=("Courier", 12),
            fg="#00FF00",
            bg="black"
        ).pack(pady=5)
        address_text = tk.Text(dialog, height=1, width=40, font=("Courier", 12), fg="#FFFFFF", bg="#001100")
        address_text.insert("1.0", wallet_data['address'])
        address_text.configure(state='disabled')
        address_text.pack(pady=5)
        tk.Label(
            dialog,
            text="Secret Seed (NEVER SHARE THIS):",
            font=("Courier", 12),
            fg="#FF0000",
            bg="black"
        ).pack(pady=5)
        seed_frame = tk.Frame(dialog, bg="black")
        seed_frame.pack(pady=5)
        seed_var = tk.StringVar(value=wallet_data['seed'])
        seed_entry = tk.Entry(seed_frame, font=("Courier", 12), fg="#FFFFFF", bg="#330000", width=40, textvariable=seed_var, show="•")
        seed_entry.pack(side="left", padx=5)
        def toggle_seed():
            current = seed_entry.cget("show")
            seed_entry.config(show="" if current == "•" else "•")
        tk.Button(seed_frame, text="Show/Hide", font=("Courier", 10), fg="#FFFFFF", bg="#550000", command=toggle_seed).pack(side="left", padx=5)
        tk.Label(dialog, text="⚠️ This wallet has 0 XRP until funded ⚠️", font=("Courier", 12), fg="#FFFF00", bg="black").pack(pady=10)
        button_frame = tk.Frame(dialog, bg="black")
        button_frame.pack(pady=15)
        def copy_to_clipboard(text, label="text"):
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            messagebox.showinfo("Copied", f"{label} copied to clipboard")
        tk.Button(button_frame, text="Copy Address", font=("Courier", 10), fg="#00FF00", bg="black", command=lambda: copy_to_clipboard(wallet_data['address'], "Address")).pack(side="left", padx=10)
        tk.Button(button_frame, text="Copy Seed", font=("Courier", 10), fg="#FF0000", bg="black", command=lambda: copy_to_clipboard(wallet_data['seed'], "Seed")).pack(side="left", padx=10)
        tk.Button(dialog, text="I HAVE SAVED MY SEED", font=("Courier", 12, "bold"), fg="#FFFFFF", bg="#003300", command=dialog.destroy).pack(pady=15)

    def open_wallet_ui(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Open Existing Wallet")
        dialog.geometry("520x380")
        dialog.configure(bg="black")
        tk.Label(dialog, text="ENTER YOUR SEED TO ACCESS WALLET", font=("Courier", 16, "bold"), fg="#00FF00", bg="black").pack(pady=20)
        warning_frame = tk.Frame(dialog, bg="#330000", bd=2)
        warning_frame.pack(fill="x", padx=20, pady=10)
        tk.Label(warning_frame, text="⚠️ MAINNET WALLET WARNING ⚠️", font=("Courier", 12, "bold"), fg="#FFFFFF", bg="#330000").pack(pady=5)
        tk.Label(warning_frame, text="• This will access a REAL wallet on mainnet\n• You will be able to send REAL XRP\n• Transactions are IRREVERSIBLE", font=("Courier", 10), fg="#FFFFFF", bg="#330000", justify="left").pack(pady=5)
        tk.Label(dialog, text="Enter your secret seed:", font=("Courier", 12), fg="#00FF00", bg="black").pack(pady=15)
        seed_frame = tk.Frame(dialog, bg="black")
        seed_frame.pack(pady=5)
        seed_var = tk.StringVar()
        seed_entry = tk.Entry(seed_frame, font=("Courier", 12), fg="#00FF00", bg="#001100", width=40, show="•", textvariable=seed_var)
        seed_entry.pack(side="left", padx=5)
        def toggle_seed():
            current = seed_entry.cget("show")
            seed_entry.config(show="" if current == "•" else "•")
        tk.Button(seed_frame, text="Show/Hide", font=("Courier", 10), fg="#FFFFFF", bg="#003300", command=toggle_seed).pack(side="left", padx=5)
        def open_wallet():
            seed = seed_var.get().strip()
            if not seed:
                messagebox.showerror("Error", "Seed cannot be empty")
                return
            try:
                wallet_data = self.wallet_manager.recover_wallet(seed)
                self.current_wallet = wallet_data
                self.update_wallet_status()
                messagebox.showinfo("Success", f"Wallet opened successfully\n\nAddress: {wallet_data['address']}\n\n⚠️ Connected to REAL MAINNET")
                dialog.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Invalid seed: {str(e)}")
        tk.Button(dialog, text="OPEN WALLET", command=open_wallet, font=("Courier", 14, "bold"), fg="#00FF00", bg="#003300").pack(pady=20)

    def save_encrypted_wallet_ui(self):
        if not self.current_wallet:
            messagebox.showerror("No Wallet", "Please open or create a wallet first")
            return
        pwd = simpledialog.askstring("Set Password", "Enter a strong password to encrypt the wallet:", show="•", parent=self.root)
        if not pwd:
            return
        try:
            self.wallet_manager.save_encrypted(self.current_wallet, pwd)
            messagebox.showinfo("Saved", f"Encrypted keystore saved to {DEFAULT_KEYSTORE_PATH}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save keystore: {e}")

    def load_encrypted_wallet_ui(self):
        pwd = simpledialog.askstring("Unlock Wallet", "Enter your keystore password:", show="•", parent=self.root)
        if not pwd:
            return
        try:
            wallet_data = self.wallet_manager.load_encrypted(pwd)
            recovered = self.wallet_manager.recover_wallet(wallet_data["seed"])
            if recovered["address"] != wallet_data["address"]:
                raise ValueError("Address/Seed mismatch in keystore.")
            self.current_wallet = recovered
            self.update_wallet_status()
            messagebox.showinfo("Loaded", f"Wallet loaded from keystore.\nAddress: {recovered['address']}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load keystore: {e}")

    def show_wallet_info(self):
        if not self.current_wallet:
            messagebox.showinfo("No Wallet", "No wallet is currently loaded")
            return
        try:
            total, spendable, owner_count = self.xrpl_client.calculate_spendable_balance(self.current_wallet['address'])
            info_text = (
                f"Address: {self.current_wallet['address']}\n\n"
                f"Balance (total): {total:.6f} XRP\n"
                f"Balance (spendable): {spendable:.6f} XRP\n"
                f"OwnerCount: {owner_count}\n"
                f"Created/Opened: {self.current_wallet.get('created_at', 'Unknown')}\n\n"
                f"Network: MAINNET (real XRP)"
            )
            messagebox.showinfo("Wallet Information", info_text)
        except Exception as e:
            messagebox.showerror("Error", f"Could not retrieve wallet info: {str(e)}")

    def update_wallet_info(self):
        """Mise à jour manuelle des informations du wallet."""
        if not self.current_wallet:
            messagebox.showinfo("No Wallet", "No wallet is currently loaded")
            return
        try:
            total, spendable, owner_count = self.xrpl_client.calculate_spendable_balance(self.current_wallet['address'])
            msg = (
                f"Wallet Updated:\n"
                f"Address: {self.current_wallet['address']}\n"
                f"Total Balance: {total:.6f} XRP\n"
                f"Spendable Balance: {spendable:.6f} XRP\n"
                f"Owner Count: {owner_count}"
            )
            messagebox.showinfo("Wallet Update", msg)
            self.update_wallet_status()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update wallet info: {str(e)}")

    def view_balance(self):
        if not self.current_wallet:
            messagebox.showerror("No Wallet", "Please open or create a wallet first")
            return
        try:
            total, spendable, _ = self.xrpl_client.calculate_spendable_balance(self.current_wallet['address'])
            price = self.xrpl_client.get_xrp_price_usd()
            total_usd = (float(total) * price) if price else None
            spendable_usd = (float(spendable) * price) if price else None
            msg = (
                f"Address: {self.current_wallet['address']}\n\n"
                f"Total: {total:.6f} XRP"
            )
            if total_usd is not None:
                msg += f" (~${total_usd:.2f})"
            msg += f"\nSpendable: {spendable:.6f} XRP"
            if spendable_usd is not None:
                msg += f" (~${spendable_usd:.2f})"
            msg += "\n\n⚠️ This is REAL money on mainnet"
            messagebox.showinfo("REAL XRP Balance", msg)
        except Exception as e:
            messagebox.showerror("Error", f"Could not retrieve balance: {str(e)}")

    def send_xrp_ui(self):
        if not self.current_wallet:
            messagebox.showerror("No Wallet", "Please open or create a wallet first")
            return
        try:
            total, spendable, _ = self.xrpl_client.calculate_spendable_balance(self.current_wallet['address'])
            if spendable <= Decimal("0"):
                messagebox.showerror("Insufficient Funds", f"Your spendable balance is 0 XRP.\nTotal: {total:.6f} XRP.\nXRPL reserve not satisfied.")
                return
        except Exception as e:
            messagebox.showerror("Error", f"Could not check balance: {str(e)}")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Send XRP - REAL TRANSACTION")
        dialog.geometry("520x520")
        dialog.configure(bg="black")
        warning_frame = tk.Frame(dialog, bg="#550000")
        warning_frame.pack(fill="x", pady=(0, 20))
        tk.Label(warning_frame, text="⚠️ WARNING: SENDING REAL XRP ⚠️", font=("Courier", 14, "bold"), fg="#FFFFFF", bg="#550000").pack(pady=10)
        tk.Label(dialog, text=f"Available (spendable): {spendable:.6f} XRP", font=("Courier", 12), fg="#FFFF00", bg="black").pack(pady=10)
        tk.Label(dialog, text="Destination Address:", font=("Courier", 12), fg="#00FF00", bg="black").pack(pady=10)
        dest_entry = tk.Entry(dialog, font=("Courier", 12), fg="#00FF00", bg="#001100", width=45)
        dest_entry.pack(pady=5)
        tk.Label(dialog, text="Destination Tag (optional, integer):", font=("Courier", 12), fg="#00FF00", bg="black").pack(pady=10)
        tag_entry = tk.Entry(dialog, font=("Courier", 12), fg="#00FF00", bg="#001100", width=20)
        tag_entry.pack(pady=5)
        tk.Label(dialog, text="Amount (XRP):", font=("Courier", 12), fg="#00FF00", bg="black").pack(pady=10)
        amount_entry = tk.Entry(dialog, font=("Courier", 12), fg="#00FF00", bg="#001100", width=20)
        amount_entry.pack(pady=5)
        def parse_amount(s: str) -> Optional[Decimal]:
            try:
                d = Decimal(s.strip())
                if d <= Decimal("0"):
                    return None
                return d.quantize(Decimal("0.000001"), rounding=ROUND_DOWN)
            except (InvalidOperation, ValueError):
                return None
        def send_xrp():
            destination = dest_entry.get().strip()
            amount_dec = parse_amount(amount_entry.get())
            if not destination:
                messagebox.showerror("Error", "Destination address cannot be empty")
                return
            if not is_valid_classic_address(destination):
                messagebox.showerror("Error", "Invalid XRPL classic address.")
                return
            if amount_dec is None:
                messagebox.showerror("Error", "Enter a valid positive amount (up to 6 decimals).")
                return
            tag_val = tag_entry.get().strip()
            tag = None
            if tag_val:
                try:
                    tag = int(tag_val)
                    if tag < 0 or tag > 2**32 - 1:
                        raise ValueError
                except Exception:
                    messagebox.showerror("Error", "Invalid Destination Tag (must be 0 .. 2^32-1).")
                    return
            _, spendable_now, _ = self.xrpl_client.calculate_spendable_balance(self.current_wallet['address'])
            if amount_dec > spendable_now:
                messagebox.showerror("Insufficient Funds", f"Amount exceeds spendable balance.\nSpendable: {spendable_now} XRP")
                return
            price = self.xrpl_client.get_xrp_price_usd()
            usd_est = (float(amount_dec) * price) if price else None
            confirm_msg = f"⚠️ FINAL CONFIRMATION ⚠️\n\nYou are about to send:\n{amount_dec} XRP"
            if usd_est is not None:
                confirm_msg += f" (~${usd_est:.2f})"
            confirm_msg += f"\n\nTo: {destination}"
            if tag is not None:
                confirm_msg += f"\nTag: {tag}"
            confirm_msg += "\n\nTHIS IS REAL MONEY AND CANNOT BE REVERSED\n\nProceed with transaction?"
            if not messagebox.askyesno("CONFIRM REAL TRANSACTION", confirm_msg):
                return
            try:
                result = self.wallet_manager.send_payment(self.current_wallet, destination, amount_dec, destination_tag=tag)
                if result.get("success"):
                    messagebox.showinfo("Transaction Successful",
                                          f"Transaction completed successfully!\n\nAmount: {amount_dec} XRP\nHash: {result.get('hash', 'Unknown')}\nLedger: {result.get('ledger', 'Unknown')}")
                    dialog.destroy()
                else:
                    messagebox.showerror("Transaction Failed", f"Transaction failed:\n\n{result.get('error', 'Unknown error')}")
            except Exception as e:
                messagebox.showerror("Error", f"Transaction error: {str(e)}")
        tk.Button(dialog, text="SEND XRP", font=("Courier", 14, "bold"), fg="#FFFFFF", bg="#550000", command=send_xrp).pack(pady=20)
        tk.Label(dialog, text="⚠️ MAINNET TRANSACTIONS ARE IRREVERSIBLE ⚠️", font=("Courier", 10), fg="#FF0000", bg="black").pack(pady=10)

    def show_history(self):
        if not self.current_wallet:
            messagebox.showerror("No Wallet", "Please open or create a wallet first")
            return
        try:
            transactions = self.wallet_manager.get_transactions(self.current_wallet['address'], 20)
            dialog = tk.Toplevel(self.root)
            dialog.title(f"Transaction History - {self.current_wallet['address']}")
            dialog.geometry("900x600")
            dialog.configure(bg="black")
            tk.Label(dialog, text="TRANSACTION HISTORY", font=("Courier", 16, "bold"), fg="#00FF00", bg="black").pack(pady=10)
            tk.Label(dialog, text=f"Address: {self.current_wallet['address']}", font=("Courier", 10), fg="#00FF00", bg="black").pack(pady=5)
            text_area = scrolledtext.ScrolledText(dialog, font=("Courier", 10), fg="#00FF00", bg="#001100", width=110, height=30)
            text_area.pack(padx=20, pady=10, fill="both", expand=True)
            if transactions:
                header = f"{'Date':<20} {'Type':<15} {'Amount':<20} {'Fee':<10} {'Result':<12} {'Validated':<10} {'Hash':<64}\n"
                text_area.insert("end", header)
                text_area.insert("end", "-" * 150 + "\n\n")
                for tx in transactions:
                    validated = "✓" if tx.get('validated', False) else "?"
                    line = (f"{tx.get('date', 'Unknown'):<20} {tx.get('type', 'Unknown'):<15} {tx.get('amount', '0'):<20} {tx.get('fee', '0'):<10} {tx.get('result', 'Unknown'):<12} {validated:<10} {tx.get('hash', 'Unknown')}\n\n")
                    text_area.insert("end", line)
            else:
                text_area.insert("end", "No transactions found for this account.\n\n")
                total = self.wallet_manager.get_balance(self.current_wallet['address'])
                if total <= 0:
                    text_area.insert("end", "This wallet appears to have 0 XRP. Fund it first to activate.\n")
            text_area.configure(state="disabled")
        except Exception as e:
            messagebox.showerror("Error", f"Could not retrieve transaction history: {str(e)}")

    def disconnect_wallet(self):
        if not self.current_wallet:
            messagebox.showinfo("No Wallet", "No wallet is currently loaded")
            return
        if messagebox.askyesno("Disconnect Wallet", "Are you sure you want to disconnect the current wallet?"):
            self.current_wallet = None
            self.update_wallet_status()
            messagebox.showinfo("Disconnected", "Wallet disconnected successfully")

# ==============================================================================
# POINT D'ENTRÉE PRINCIPAL
# ==============================================================================
def main():
    try:
        print(f"🚀 Starting {APP_NAME} Wallet v{APP_VERSION} - MAINNET EDITION")
        print("⚠️ WARNING: This application connects to the XRP Ledger mainnet")
        print("⚠️ WARNING: All transactions involve REAL MONEY")
        root = tk.Tk()
        app = EscapeWallet(root)
        def on_closing():
            if messagebox.askyesno("Quit", "Are you sure you want to exit?"):
                print("👋 Shutting down ESCAPE Wallet")
                root.destroy()
        root.protocol("WM_DELETE_WINDOW", on_closing)
        root.mainloop()
    except Exception as e:
        print(f"💥 Critical error: {e}")
        logger.critical(f"Critical error: {e}", exc_info=True)

if __name__ == "__main__":
    main()
