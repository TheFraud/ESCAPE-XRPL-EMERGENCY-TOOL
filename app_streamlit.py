#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESCAPE - XRP Ledger Emergency Access Tool (Streamlit Edition)
MAINNET ONLY VERSION - 100% REAL XRP
Version: 1.2.0 - Web Interface

Complete Streamlit conversion with full compatibility
"""

import os
import sys
import base64
import json
import time
import logging
from datetime import datetime
from itertools import cycle
from decimal import Decimal, InvalidOperation
from typing import Dict, List, Any, Optional, Tuple

import streamlit as st
import requests

# Logging configuration
LOG_FILE = os.path.expanduser('~/.escape_streamlit.log')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ESCAPE_STREAMLIT")

# Page config (modern Streamlit)
try:
    st.set_page_config(
        page_title="ESCAPE XRPL Emergency Tool",
        page_icon="💸",
        layout="wide",
        initial_sidebar_state="expanded"
    )
except:
    pass  # Fallback for older versions

# Import dependencies
try:
    from xrpl.clients import JsonRpcClient
    from xrpl.wallet import Wallet
    from xrpl.models.requests import AccountInfo, ServerInfo, AccountTx
    from xrpl.models.transactions import Payment
    from xrpl.utils import xrp_to_drops, drops_to_xrp
    from xrpl.transaction import submit_and_wait, autofill
    from xrpl.core.addresscodec import is_valid_classic_address
    
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.backends import default_backend
    from cryptography.fernet import Fernet
except ImportError as e:
    st.error(f"Critical import error: {e}")
    st.error("Please install: pip install streamlit xrpl-py cryptography requests")
    st.stop()

# Constants
APP_VERSION = "1.2.0"
APP_NAME = "ESCAPE XRPL Emergency Tool"
MAINNET_ENDPOINTS = [
    "https://xrplcluster.com/",
    "https://s1.ripple.com:51234/",
    "https://s2.ripple.com:51234/"
]
DEFAULT_KEYSTORE_PATH = os.path.expanduser("~/.escape_mainnet.wallet")

# Encryption functions
def generate_key(password: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=100000, backend=default_backend())
    return base64.urlsafe_b64encode(kdf.derive(password.encode()))

def encrypt_data(data: str, password: str) -> bytes:
    salt = os.urandom(16)
    key = generate_key(password, salt)
    fernet = Fernet(key)
    encrypted = fernet.encrypt(data.encode())
    return b"ESCAPE1" + salt + encrypted

def decrypt_data(encrypted_data: bytes, password: str) -> str:
    if not encrypted_data.startswith(b"ESCAPE1"):
        raise ValueError("Invalid ESCAPE keystore file.")
    encrypted_data = encrypted_data[len(b"ESCAPE1"):]
    salt = encrypted_data[:16]
    actual_encrypted = encrypted_data[16:]
    key = generate_key(password, salt)
    fernet = Fernet(key)
    return fernet.decrypt(actual_encrypted).decode()

# Secure storage class
class SecureStore:
    def __init__(self, path: Optional[str] = None):
        self.path = path or DEFAULT_KEYSTORE_PATH

    def save(self, wallet: dict, password: str) -> bool:
        if not wallet or 'seed' not in wallet or 'address' not in wallet:
            raise ValueError("Invalid wallet data.")
        payload = {"version": 1, "created_at": datetime.now().isoformat(), "address": wallet['address'], "seed": wallet['seed']}
        data = json.dumps(payload)
        encrypted = encrypt_data(data, password)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "wb") as f:
            f.write(encrypted)
        return True

    def load(self, password: str) -> dict:
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"No keystore found: {self.path}")
        with open(self.path, "rb") as f:
            encrypted = f.read()
        data = decrypt_data(encrypted, password)
        payload = json.loads(data)
        if "seed" not in payload or "address" not in payload:
            raise ValueError("Corrupted keystore.")
        return payload

# XRPL Client with fixed API handling
class MainnetClient:
    def __init__(self):
        self.http_endpoints = MAINNET_ENDPOINTS.copy()
        self.endpoint_cycle = cycle(self.http_endpoints)
        self.current_endpoint = next(self.endpoint_cycle)
        self.client = JsonRpcClient(self.current_endpoint)
        self.connection_status = False
        self.connection_info = {}
        self.connection_latency = 0
        self.failed_attempts = 0
        self._xrp_price_usd = None
        self._last_price_fetch = 0

    def switch_endpoint(self):
        self.current_endpoint = next(self.endpoint_cycle)
        logger.info(f"Switching to: {self.current_endpoint}")
        self.client = JsonRpcClient(self.current_endpoint)
        self.failed_attempts = 0

    def _safe_get_result(self, response) -> dict:
        """Safely extract result from XRPL response"""
        if hasattr(response, 'result'):
            result = response.result
            if hasattr(result, 'to_dict') and callable(getattr(result, 'to_dict')):
                try:
                    return result.to_dict()
                except:
                    return result if isinstance(result, dict) else {}
            else:
                return result if isinstance(result, dict) else {}
        return {}

    def check_connection(self) -> Tuple[bool, Dict]:
        try:
            start_time = time.time()
            response = self.client.request(ServerInfo())
            if not response.is_successful():
                raise ConnectionError("ServerInfo request failed")
            
            latency = round((time.time() - start_time) * 1000)
            self.connection_latency = latency
            result = self._safe_get_result(response)
            self.connection_info = result.get("info", {})
            self.connection_status = True
            self.failed_attempts = 0
            
            return True, {
                "state": self.connection_info.get("server_state", "UNKNOWN"),
                "version": self.connection_info.get("build_version", "UNKNOWN"),
                "latency": latency,
                "endpoint": self.current_endpoint
            }
        except Exception as e:
            logger.error(f"Connection error: {e}")
            self.connection_status = False
            self.failed_attempts += 1
            if self.failed_attempts >= 3:
                self.switch_endpoint()
            return False, {"error": str(e), "endpoint": self.current_endpoint}

    def get_account_info(self, address: str) -> Optional[dict]:
        try:
            response = self.client.request(AccountInfo(account=address, ledger_index="validated", strict=True))
            if response.is_successful():
                result = self._safe_get_result(response)
                return result.get("account_data")
        except Exception as e:
            logger.error(f"Error in get_account_info: {e}")
        return None

    def get_account_balance(self, address: str) -> Decimal:
        try:
            account_info = self.get_account_info(address)
            if not account_info:
                return Decimal("0")
            balance_drops = account_info.get("Balance", "0")
            return Decimal(str(drops_to_xrp(balance_drops)))
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return Decimal("0")

    def get_reserve_params(self) -> Tuple[Decimal, Decimal]:
        try:
            response = self.client.request(ServerInfo())
            if response.is_successful():
                result = self._safe_get_result(response)
                info = result.get("info", {})
                validated_ledger = info.get("validated_ledger", {})
                base = Decimal(str(validated_ledger.get("reserve_base_xrp", "10")))
                inc = Decimal(str(validated_ledger.get("reserve_inc_xrp", "2")))
                return base, inc
        except Exception as e:
            logger.warning(f"Error getting reserve params: {e}")
        return Decimal("10"), Decimal("2")

    def calculate_spendable_balance(self, address: str) -> Tuple[Decimal, Decimal, int]:
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
        try:
            now = time.time()
            if self._xrp_price_usd is not None and (now - self._last_price_fetch) < 60:
                return self._xrp_price_usd
            
            response = requests.get("https://api.coingecko.com/api/v3/simple/price", 
                                   params={"ids": "ripple", "vs_currencies": "usd"}, timeout=5)
            if response.ok:
                data = response.json()
                price = float(data.get("ripple", {}).get("usd", 0))
                if price > 0:
                    self._xrp_price_usd = price
                    self._last_price_fetch = now
                    return price
        except Exception as e:
            logger.warning(f"Price fetch failed: {e}")
        return None

    def send_xrp(self, sender_wallet: dict, destination: str, amount_xrp: Decimal, destination_tag: Optional[int] = None) -> dict:
        try:
            if not is_valid_classic_address(destination):
                return {"success": False, "error": "Invalid XRPL address."}
            if amount_xrp <= Decimal("0"):
                return {"success": False, "error": "Amount must be greater than 0."}
            
            total, spendable, _ = self.calculate_spendable_balance(sender_wallet['address'])
            if amount_xrp > spendable:
                return {"success": False, "error": f"Insufficient funds. Spendable: {spendable} XRP, Total: {total} XRP"}
            
            drops_amount = xrp_to_drops(str(amount_xrp))
            tx_kwargs = {"account": sender_wallet['address'], "destination": destination, "amount": drops_amount}
            
            if destination_tag is not None:
                if not isinstance(destination_tag, int) or not (0 <= destination_tag < 2**32):
                    return {"success": False, "error": "Invalid Destination Tag."}
                tx_kwargs["destination_tag"] = destination_tag
            
            payment = Payment(**tx_kwargs)
            payment = autofill(self.client, payment)
            wallet_obj = Wallet(seed=sender_wallet['seed'])
            
            logger.info(f"Sending {amount_xrp} XRP to {destination}")
            response = submit_and_wait(payment, self.client, wallet_obj)
            
            if response.is_successful():
                result = self._safe_get_result(response)
                tx_hash = result.get("hash") or result.get("tx_json", {}).get("hash")
                return {"success": True, "hash": tx_hash, "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            else:
                error = result.get("engine_result_message", "Unknown error")
                return {"success": False, "error": error}
        except Exception as e:
            logger.error(f"Error sending XRP: {e}")
            return {"success": False, "error": str(e)}

    def get_transaction_history(self, address: str, limit: int = 10) -> List[Dict]:
        try:
            response = self.client.request(AccountTx(account=address, ledger_index_min=-1, ledger_index_max=-1, limit=limit, binary=False))
            if not response.is_successful():
                return []
            
            result = self._safe_get_result(response)
            transactions = []
            
            for tx_info in result.get("transactions", []):
                tx = tx_info.get("tx", {})
                meta = tx_info.get("meta", {})
                
                date_str = "Unknown"
                if "date" in tx:
                    try:
                        timestamp = tx["date"] + 946684800
                        date_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        pass
                
                amount = "0"
                if "Amount" in tx:
                    try:
                        if isinstance(tx["Amount"], str):
                            amount = f"{drops_to_xrp(tx['Amount'])} XRP"
                        else:
                            amount = f"{tx['Amount']['value']} {tx['Amount']['currency']}"
                    except:
                        amount = str(tx.get("Amount", "0"))
                
                transactions.append({
                    "hash": tx.get("hash", "Unknown"),
                    "date": date_str,
                    "type": tx.get("TransactionType", "Unknown"),
                    "amount": amount,
                    "fee": f"{drops_to_xrp(tx.get('Fee', '0'))} XRP",
                    "result": meta.get("TransactionResult", "Unknown"),
                    "validated": tx_info.get("validated", False)
                })
            
            return transactions
        except Exception as e:
            logger.error(f"Error getting transaction history: {e}")
            return []

# Wallet Manager
class WalletManager:
    def __init__(self, xrpl_client: MainnetClient):
        self.xrpl_client = xrpl_client
        self.keystore = SecureStore()

    def create_wallet(self) -> dict:
        wallet = Wallet.create()
        return {
            'address': wallet.classic_address,
            'seed': wallet.seed,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    def recover_wallet(self, seed: str) -> dict:
        try:
            wallet = Wallet(seed=seed.strip())
            return {'address': wallet.classic_address, 'seed': seed.strip(), 'created_at': 'Recovered wallet'}
        except Exception as e:
            raise ValueError(f"Invalid seed: {e}")

    def get_balance(self, address: str) -> Decimal:
        return self.xrpl_client.get_account_balance(address)

    def get_spendable(self, address: str) -> Tuple[Decimal, Decimal, int]:
        return self.xrpl_client.calculate_spendable_balance(address)

    def send_payment(self, wallet: dict, destination: str, amount: Decimal, destination_tag: Optional[int] = None) -> dict:
        return self.xrpl_client.send_xrp(wallet, destination, amount, destination_tag=destination_tag)

    def get_transactions(self, address: str, limit: int = 20) -> List[Dict]:
        return self.xrpl_client.get_transaction_history(address, limit)

    def save_encrypted(self, wallet: dict, password: str) -> bool:
        return self.keystore.save(wallet, password)

    def load_encrypted(self, password: str) -> dict:
        return self.keystore.load(password)

# Initialize session state
def init_session_state():
    if "wallet" not in st.session_state:
        st.session_state.wallet = None
    if "xrpl_client" not in st.session_state:
        st.session_state.xrpl_client = MainnetClient()
    if "wallet_manager" not in st.session_state:
        st.session_state.wallet_manager = WalletManager(st.session_state.xrpl_client)

# Main application
def main():
    # Custom CSS
    st.markdown("""
    <style>
    .big-font { font-size:30px !important; color: #00FF00; font-family: monospace; text-align: center; }
    .warning-red { color: #FF0000; font-weight: bold; text-align: center; }
    </style>
    """, unsafe_allow_html=True)
    
    init_session_state()
    
    # Header
    st.markdown('<p class="big-font">🚨 ESCAPE XRPL Emergency Tool v1.2.0 🚨</p>', unsafe_allow_html=True)
    st.markdown('<p class="warning-red">⚠️ MAINNET - REAL XRP - TRANSACTIONS ARE IRREVERSIBLE ⚠️</p>', unsafe_allow_html=True)
    
    # Network Status
    st.subheader("🌐 Network Status")
    with st.spinner("Checking connection..."):
        connected, info = st.session_state.xrpl_client.check_connection()
    
    if connected:
        st.success(f"🟢 Connected to {info.get('endpoint', 'Unknown')}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Latency", f"{info.get('latency', 0)} ms")
        col2.metric("Server State", info.get('state', 'Unknown'))
        col3.metric("Version", info.get('version', 'Unknown')[:20])
    else:
        st.error(f"🔴 Connection Failed: {info.get('error', 'Unknown')}")
    
    st.markdown("---")
    
    # Wallet Status
    st.subheader("💼 Wallet Status")
    wallet = st.session_state.wallet
    
    if wallet:
        st.success(f"🔓 Wallet Active: {wallet['address'][:15]}...{wallet['address'][-15:]}")
        try:
            total, spendable, owner_count = st.session_state.wallet_manager.get_spendable(wallet['address'])
            price = st.session_state.xrpl_client.get_xrp_price_usd()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Balance", f"{total:.6f} XRP")
            col2.metric("Spendable", f"{spendable:.6f} XRP")
            col3.metric("Owner Count", str(owner_count))
            
            if price:
                st.info(f"💵 XRP Price: ${price:.4f} | Total Value: ~${float(total) * price:.2f}")
        except Exception as e:
            st.error(f"Error fetching balance: {e}")
    else:
        st.warning("🔒 No wallet loaded")
    
    st.markdown("---")
    
    # Sidebar Navigation
    st.sidebar.title("🚀 ESCAPE Navigation")
    menu = st.sidebar.radio("Select Action:", [
        "🏠 Home",
        "🆕 Create Wallet", 
        "🔓 Open Wallet",
        "💼 Wallet Info",
        "💸 Send XRP",
        "📜 Transaction History",
        "🔐 Encrypted Storage"
    ])
    
    # Emergency disconnect
    if wallet and st.sidebar.button("🔒 Emergency Disconnect", help="Disconnect wallet for security"):
        st.session_state.wallet = None
        st.success("Wallet disconnected")
        st.experimental_rerun()
    
    # Page content based on selection
    if menu == "🏠 Home":
        st.header("🏠 Welcome to ESCAPE")
        st.info("""
        **ESCAPE** is an emergency access tool for the XRP Ledger mainnet.
        
        **Features:**
        - Create new XRPL wallets
        - Access existing wallets with seed
        - Send real XRP transactions
        - View transaction history
        - Encrypted wallet storage
        
        **⚠️ Critical Warnings:**
        - Connects to REAL XRP Ledger mainnet
        - All transactions use REAL XRP and are IRREVERSIBLE
        - Always verify addresses before sending
        - Keep your seed phrase secure and private
        """)
        
        if not wallet:
            st.warning("👆 Use the sidebar to create or open a wallet to get started.")
    
    elif menu == "🆕 Create Wallet":
        st.header("🆕 Create New Wallet")
        st.error("🚨 WARNING: REAL MAINNET WALLET")
        st.warning("""
        - Creates wallet on REAL XRP Ledger mainnet
        - Wallet will have 0 XRP until funded
        - Need at least 10 XRP to activate
        - All transactions are IRREVERSIBLE
        """)
        
        if st.button("CREATE WALLET", type="primary"):
            try:
                wallet_data = st.session_state.wallet_manager.create_wallet()
                st.session_state.wallet = wallet_data
                st.success("✅ Wallet created successfully!")
                
                st.subheader("🔐 CRITICAL: Save Your Secret Information")
                st.info("**Address (Public - Safe to Share):**")
                st.code(wallet_data['address'])
                st.error("**Secret Seed (NEVER SHARE - Save Securely):**")
                st.code(wallet_data['seed'])
                st.warning("⚠️ This wallet has 0 XRP until funded. Send at least 10 XRP to activate.")
            except Exception as e:
                st.error(f"Failed to create wallet: {e}")
    
    elif menu == "🔓 Open Wallet":
        st.header("🔓 Open Existing Wallet")
        st.error("⚠️ MAINNET WALLET WARNING")
        st.warning("""
        - Accesses REAL wallet on mainnet
        - You can send REAL XRP
        - Transactions are IRREVERSIBLE
        """)
        
        seed_input = st.text_input("Enter your secret seed:", type="password", placeholder="sXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        
        if st.button("OPEN WALLET", type="primary"):
            if not seed_input:
                st.error("Seed cannot be empty")
            else:
                try:
                    wallet_data = st.session_state.wallet_manager.recover_wallet(seed_input)
                    st.session_state.wallet = wallet_data
                    st.success("✅ Wallet opened successfully!")
                    st.info(f"**Address:** {wallet_data['address']}")
                    st.error("🔴 Connected to REAL MAINNET")
                except Exception as e:
                    st.error(f"Invalid seed: {e}")
    
    elif menu == "💼 Wallet Info":
        st.header("💼 Wallet Information")
        if not wallet:
            st.warning("No wallet loaded. Create or open a wallet first.")
        else:
            try:
                total, spendable, owner_count = st.session_state.wallet_manager.get_spendable(wallet['address'])
                price = st.session_state.xrpl_client.get_xrp_price_usd()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("🏦 Wallet Details")
                    st.text_area("Address:", wallet['address'], height=60, disabled=True)
                    st.info(f"Created/Opened: {wallet.get('created_at', 'Unknown')}")
                
                with col2:
                    st.subheader("💰 Balance Information")
                    st.metric("Total Balance", f"{total:.6f} XRP")
                    st.metric("Spendable Balance", f"{spendable:.6f} XRP")
                    st.metric("Owner Count", str(owner_count))
                
                if price:
                    st.subheader("💵 USD Values")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("XRP Price", f"${price:.4f}")
                    col2.metric("Total Value", f"${float(total) * price:.2f}")
                    col3.metric("Spendable Value", f"${float(spendable) * price:.2f}")
            except Exception as e:
                st.error(f"Could not retrieve wallet information: {e}")
    
    elif menu == "💸 Send XRP":
        st.header("💸 Send XRP")
        if not wallet:
            st.error("No wallet loaded. Create or open a wallet first.")
        else:
            try:
                total, spendable, _ = st.session_state.wallet_manager.get_spendable(wallet['address'])
                
                if spendable <= Decimal("0"):
                    st.error("⚠️ Insufficient spendable funds!")
                    st.info(f"Total: {total:.6f} XRP, but all is reserved.")
                else:
                    st.success(f"💰 Available to send: {spendable:.6f} XRP")
                    
                    destination = st.text_input("Destination Address:", placeholder="rXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
                    amount_input = st.text_input("Amount (XRP):", placeholder="0.000001")
                    destination_tag = st.text_input("Destination Tag (optional):", placeholder="12345")
                    
                    if destination and amount_input:
                        if not is_valid_classic_address(destination):
                            st.error("❌ Invalid XRPL address format")
                        else:
                            try:
                                amount = Decimal(amount_input.strip())
                                if amount <= Decimal("0"):
                                    st.error("❌ Amount must be greater than 0")
                                elif amount > spendable:
                                    st.error(f"❌ Amount exceeds spendable balance ({spendable:.6f} XRP)")
                                else:
                                    tag = None
                                    if destination_tag.strip():
                                        try:
                                            tag = int(destination_tag.strip())
                                        except ValueError:
                                            st.error("❌ Destination tag must be a number")
                                            tag = "invalid"
                                    
                                    if tag != "invalid":
                                        st.error("🚨 WARNING: REAL MAINNET TRANSACTION")
                                        st.warning("Transactions are IRREVERSIBLE. Double-check all details!")
                                        
                                        confirm = st.checkbox("I confirm this transaction uses REAL XRP and is IRREVERSIBLE")
                                        
                                        if confirm and st.button("🚀 SEND XRP", type="primary"):
                                            with st.spinner("Sending transaction..."):
                                                result = st.session_state.wallet_manager.send_payment(wallet, destination, amount, destination_tag=tag)
                                            
                                            if result.get("success"):
                                                st.success("✅ Transaction sent successfully!")
                                                st.write(f"**TX Hash:** {result.get('hash', 'Unknown')}")
                                                st.write(f"**Date:** {result.get('date', 'Unknown')}")
                                            else:
                                                st.error(f"❌ Transaction failed: {result.get('error', 'Unknown error')}")
                            except (InvalidOperation, ValueError):
                                st.error("❌ Invalid amount format")
            except Exception as e:
                st.error(f"Error checking balance: {e}")
    
    elif menu == "📜 Transaction History":
        st.header("📜 Transaction History")
        if not wallet:
            st.error("No wallet loaded. Create or open a wallet first.")
        else:
            st.info(f"Address: {wallet['address']}")
            
            limit = st.selectbox("Number of transactions:", [10, 20, 50, 100], index=1)
            
            if st.button("🔄 Load History"):
                with st.spinner("Loading transaction history..."):
                    transactions = st.session_state.wallet_manager.get_transactions(wallet['address'], limit)
                
                if not transactions:
                    st.warning("No transactions found for this wallet.")
                    balance = st.session_state.wallet_manager.get_balance(wallet['address'])
                    if balance <= 0:
                        st.info("💡 This wallet has 0 XRP. Fund it first to see transactions.")
                else:
                    st.success(f"Found {len(transactions)} transactions")
                    
                    for tx in transactions:
                        with st.expander(f"{tx.get('type', 'Unknown')} - {tx.get('amount', 'Unknown')} - {tx.get('date', 'Unknown')}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Hash:** {tx.get('hash', 'Unknown')}")
                                st.write(f"**Type:** {tx.get('type', 'Unknown')}")
                                st.write(f"**Amount:** {tx.get('amount', 'Unknown')}")
                                st.write(f"**Fee:** {tx.get('fee', 'Unknown')}")
                            with col2:
                                st.write(f"**Date:** {tx.get('date', 'Unknown')}")
                                st.write(f"**Result:** {tx.get('result', 'Unknown')}")
                                validated = "✅ Yes" if tx.get('validated', False) else "⏳ Pending"
                                st.write(f"**Validated:** {validated}")
    
    elif menu == "🔐 Encrypted Storage":
        st.header("🔐 Encrypted Wallet Storage")
        
        tab1, tab2 = st.tabs(["💾 Save Encrypted", "📂 Load Encrypted"])
        
        with tab1:
            st.subheader("Save Current Wallet")
            if not wallet:
                st.warning("No wallet loaded to save.")
            else:
                st.success(f"Current wallet: {wallet['address'][:10]}...{wallet['address'][-10:]}")
                
                password = st.text_input("Set encryption password:", type="password")
                confirm_password = st.text_input("Confirm password:", type="password")
                
                if password and confirm_password:
                    if password != confirm_password:
                        st.error("Passwords do not match")
                    elif len(password) < 8:
                        st.warning("Password should be at least 8 characters")
                    else:
                        if st.button("💾 Save Encrypted Wallet"):
                            try:
                                st.session_state.wallet_manager.save_encrypted(wallet, password)
                                st.success(f"✅ Wallet saved encrypted to {DEFAULT_KEYSTORE_PATH}")
                            except Exception as e:
                                st.error(f"Failed to save wallet: {e}")
        
        with tab2:
            st.subheader("Load Encrypted Wallet")
            
            if not os.path.exists(DEFAULT_KEYSTORE_PATH):
                st.warning("No encrypted wallet found on this device.")
            else:
                st.info(f"Found encrypted wallet at: {DEFAULT_KEYSTORE_PATH}")
                
                password = st.text_input("Enter wallet password:", type="password", key="load_password")
                
                if st.button("🔓 Load Encrypted Wallet"):
                    if not password:
                        st.error("Password cannot be empty")
                    else:
                        try:
                            wallet_data = st.session_state.wallet_manager.load_encrypted(password)
                            recovered = st.session_state.wallet_manager.recover_wallet(wallet_data["seed"])
                            st.session_state.wallet = recovered
                            st.success("✅ Wallet loaded successfully from keystore!")
                            st.info(f"Address: {recovered['address']}")
                        except Exception as e:
                            st.error(f"Failed to load wallet: {e}")

if __name__ == "__main__":
    main()
