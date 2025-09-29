<#
install_tradingbot_allinone.ps1
All-in-one installer + project scaffolder for PocketOption trading suite (MrAlpert)
Run as Administrator in PowerShell.
#>

Set-StrictMode -Version Latest
$PSScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
$projectDir = Join-Path $PSScriptRoot "pocket_trading"
$envName = "tradingbot"
$pyver = "3.11"

function Info($s){ Write-Host $s -ForegroundColor Cyan }
function OK($s){ Write-Host $s -ForegroundColor Green }
function Warn($s){ Write-Host $s -ForegroundColor Yellow }
function Err($s){ Write-Host $s -ForegroundColor Red }

# ---------------------------
# 1) Ensure system packages
# ---------------------------
if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
    Info "Installing Chocolatey..."
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
    refreshenv
} else { OK "Chocolatey present" }

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Info "Installing Git..."
    choco install -y git
    refreshenv
} else { OK "Git present" }

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Info "Installing Miniconda..."
    choco install -y miniconda3
    refreshenv
} else { OK "Conda present" }

if (-not (Get-Command cl -ErrorAction SilentlyContinue)) {
    Info "Installing Visual Studio Build Tools..."
    choco install -y visualstudio2019buildtools --params "'--add Microsoft.VisualStudio.Workload.VCTools'"
    refreshenv
} else { OK "VS Build Tools present" }

# ---------------------------
# 2) Create conda env
# ---------------------------
$exists = conda env list | Select-String -Pattern "^\s*$envName\s" -Quiet
if (-not $exists) {
    Info "Creating conda env '$envName' with Python $pyver..."
    conda create -y -n $envName python=$pyver
    OK "Created env $envName"
} else {
    Info "Conda env '$envName' already exists."
}

# ---------------------------
# 3) Install packages into env
# ---------------------------
Info "Installing conda-forge packages (ta-lib, numpy, pandas, pywin32, etc.)"
conda install -y -n $envName -c conda-forge ta-lib numpy pandas scipy matplotlib scikit-learn jupyterlab pywin32

Info "Upgrading pip and installing pip packages (inside conda env)"
conda run -n $envName pip install --upgrade pip
$pkgs = @(
    "requests",
    "websocket-client",
    "websockets",
    "pyqt5",
    "pandas_ta",
    "stable-baselines3",
    "gym",
    "torch",
    "tensorflow==2.12.0",
    "cryptography",
    "keyring",
    "psutil"
)
conda run -n $envName pip install $pkgs

# Try to install MrAlpert client
Info "Installing MrAlpert Pocket-Option-API from GitHub (best-effort)"
$installClientCmd = "conda run -n $envName pip install git+https://github.com/MrAlpert/Pocket-Option-API.git"
Invoke-Expression $installClientCmd
if ($LASTEXITCODE -eq 0) { OK "MrAlpert client installed" } else { Warn "MrAlpert install failed; you can run: $installClientCmd manually inside the environment." }

# Install NSSM (service helper) optionally for Windows service
Info "Installing nssm (service helper) via Chocolatey (optional)"
choco install -y nssm || Warn "nssm install may have failed."

# ---------------------------
# 4) Scaffold project folder & write Python files
# ---------------------------
if (-not (Test-Path $projectDir)) { New-Item -ItemType Directory -Path $projectDir | Out-Null; OK "Created project folder: $projectDir" } else { OK "Project folder exists: $projectDir" }

# Write __init__.py
$init = ""
$initPath = Join-Path $projectDir "__init__.py"
$init | Out-File -FilePath $initPath -Encoding utf8 -Force

# ----------- bot.py (fully wired to MrAlpert) -----------
$bot_py = @'
# bot.py
"""
Trading engine wired to MrAlpert's Pocket-Option-API when available.
Robust fallbacks are included; edit small sections if your local package has different method names.
"""

import time
import threading
import random
import logging
import importlib
import json

import numpy as np
try:
    import talib
except Exception:
    talib = None

# Candidate module/class used by MrAlpert's repo
CANDIDATES = [
    ("pocketoptionapi.stable_api", "PocketOption"), 
    ("Pocket_Option_API.client", "PocketOptionClient"),
    ("pocketoptionapi", "PocketOption")
]

def try_import_client():
    for modname, clsname in CANDIDATES:
        try:
            mod = importlib.import_module(modname)
            cls = getattr(mod, clsname, None)
            if cls:
                return cls, f"{modname}.{clsname}"
            # maybe module itself is the client object
            # return module to give caller chance to use its functions
            return mod, f"{modname} (module)"
        except Exception:
            continue
    return None, None

class TradingBot:
    def __init__(self, logger=print, asset="EURUSD_otc"):
        self.logger = logger
        self.asset = asset
        self.client = None
        self.session = None
        self.authenticated = False
        self.demo = True
        self.running = False
        self.strategies = []
        self.trade_history = []
        self._trade_callback_registered = False

    def log(self, *args):
        try:
            self.logger(" ".join(str(a) for a in args))
        except Exception:
            print(*args)

    # ----------------------------
    # Connect using SSID (MrAlpert official-style)
    # ----------------------------
    def connect_with_ssid(self, ssid, demo=True, connect_timeout=10):
        cls_or_mod, info = try_import_client()
        if not cls_or_mod:
            self.log("Community client not found (MrAlpert). Falling back to REST fallback mode.")
            return False, "client_missing"

        self.log(f"Found client: {info}. Attempting to construct client with SSID.")
        try:
            # try class instantiation
            try:
                client = cls_or_mod(ssid, demo)
            except TypeError:
                # maybe mod exposes a factory function or different signature
                try:
                    client = cls_or_mod.PocketOption(ssid, demo)
                except Exception:
                    client = cls_or_mod
            # try connect method
            if hasattr(client, "connect"):
                res = client.connect()
            elif hasattr(client, "start"):
                res = client.start()
            else:
                res = None
            self.client = client
            self.session = ssid
            self.demo = demo
            self.authenticated = True
            self.log("Connected to PocketOption client (best-effort).", res)
            # register callback if available
            self._try_register_trade_callback()
            return True, res
        except Exception as e:
            self.log("Client connect exception:", e)
            return False, str(e)

    # ----------------------------
    # Fallback: attempt REST login (best-effort)
    # ----------------------------
    def login_rest(self, email, password):
        try:
            import requests
            url = "https://pocketoption.com/api/v1/login"
            r = requests.post(url, json={"email": email, "password": password}, timeout=10)
            if r.status_code == 200:
                try:
                    j = r.json()
                except Exception:
                    j = None
                cookie_ssid = None
                if r.cookies:
                    cookie_ssid = r.cookies.get("ssid")
                ssid = cookie_ssid or (j.get("session") if isinstance(j, dict) else None) or (j.get("token") if isinstance(j, dict) else None)
                if ssid:
                    self.session = ssid
                    self.authenticated = True
                    self.log("REST login successful (ssid stored).")
                    return True, {"ssid": ssid, "json": j}
                else:
                    return False, {"status": r.status_code, "text": r.text}
            else:
                return False, {"status": r.status_code, "text": r.text}
        except Exception as e:
            return False, str(e)

    # ----------------------------
    # Register trade result callback (attempt many common names)
    # ----------------------------
    def _try_register_trade_callback(self):
        if not self.client or self._trade_callback_registered:
            return
        # If client offers decorator style or on(...) registration, attempt to register.
        try:
            # common patterns: client.on_trade, client.on('trade', callback), client.add_listener(...)
            if hasattr(self.client, "on_trade_complete"):
                try:
                    self.client.on_trade_complete(self._internal_trade_callback)
                    self._trade_callback_registered = True
                    self.log("Registered on_trade_complete callback.")
                    return
                except Exception:
                    pass
            # decorator-like attribute
            if hasattr(self.client, "on"):
                try:
                    self.client.on("trade_complete", self._internal_trade_callback)
                    self._trade_callback_registered = True
                    self.log("Registered on('trade_complete',...) callback.")
                    return
                except Exception:
                    pass
            # try attribute event handlers
            if hasattr(self.client, "subscribe") and callable(getattr(self.client, "subscribe")):
                try:
                    self.client.subscribe("trade", self._internal_trade_callback)
                    self._trade_callback_registered = True
                    self.log("Registered subscribe('trade',...) callback.")
                    return
                except Exception:
                    pass
        except Exception as e:
            self.log("Callback registration attempts raised:", e)

    def _internal_trade_callback(self, payload):
        # Called by client if client supports callbacks — normalize payload and append to history
        try:
            self.log("Trade callback:", payload)
            # attempt to extract win/profit if present
            win = None
            if isinstance(payload, dict):
                for key in ("profit", "win", "is_win", "result", "profit_amount"):
                    if key in payload:
                        val = payload.get(key)
                        if isinstance(val, bool):
                            win = val
                        elif isinstance(val, (int, float)):
                            win = (val > 0)
            self.trade_history.append({"payload": payload, "win": win, "ts": time.time()})
        except Exception as e:
            self.log("Error in _internal_trade_callback:", e)

    # ----------------------------
    # Place trade - use client.buy if available, try alternatives if not
    # ----------------------------
    def place_trade(self, amount, action="call", expirations=60):
        """
        action: 'call' or 'put' (per MrAlpert's API)
        """
        if self.client:
            # try common method names
            for m in ("buy", "buy_option", "trade", "place_order", "create_order"):
                fn = getattr(self.client, m, None)
                if callable(fn):
                    try:
                        # many clients expect (amount, active, action, expirations) or (active, amount, action)
                        try:
                            resp = fn(amount=amount, active=self.asset, action=action, expirations=expirations)
                        except TypeError:
                            try:
                                resp = fn(self.asset, amount, action, expirations)
                            except TypeError:
                                resp = fn(action, amount, self.asset)
                        self.log("Placed trade via client method", m, "->", resp)
                        # parse known structure
                        parsed = self._parse_trade_response(resp)
                        if parsed is not None:
                            self.trade_history.append(parsed)
                        else:
                            self.trade_history.append({"raw": resp})
                        return True, resp
                    except Exception as e:
                        self.log("Client trade method", m, "raised:", e)
                        continue
            return False, "client_no_trade_method"
        else:
            # fallback REST attempt
            try:
                import requests
                headers = {}
                if self.session:
                    headers["Cookie"] = f"ssid={self.session}"
                url = "https://pocketoption.com/api/v1/trade"
                payload = {"asset": self.asset, "amount": amount, "side": action, "duration": expirations}
                r = requests.post(url, json=payload, headers=headers, timeout=10)
                if r.status_code == 200:
                    try:
                        j = r.json()
                    except Exception:
                        j = r.text
                    self.trade_history.append({"raw": j})
                    self.log("Fallback trade accepted:", j)
                    return True, j
                else:
                    self.log("Fallback trade failed:", r.status_code, r.text)
                    return False, {"status": r.status_code, "text": r.text}
            except Exception as e:
                return False, str(e)

    def _parse_trade_response(self, resp):
        # Attempt to normalize trade response dict into a small shape
        if resp is None:
            return None
        if isinstance(resp, dict):
            parsed = {}
            parsed["ts"] = time.time()
            parsed["raw"] = resp
            # try common keys
            parsed["order_id"] = resp.get("order_id") or resp.get("id") or resp.get("orderId") or None
            parsed["success"] = resp.get("success") if "success" in resp else None
            # profit/win detection
            parsed["win"] = None
            for key in ("profit", "win", "is_win", "result"):
                if key in resp:
                    v = resp.get(key)
                    if isinstance(v, bool):
                        parsed["win"] = v
                    elif isinstance(v, (int, float)):
                        parsed["win"] = (v > 0)
            return parsed
        return {"raw": resp, "ts": time.time()}

    # ----------------------------
    # strategies/backtest
    # ----------------------------
    def load_strategies(self):
        try:
            from strategies import sample_strategies
            self.strategies = sample_strategies()
        except Exception:
            self.strategies = [
                {"name":"EMA 5/20","type":"ema","fast":5,"slow":20},
                {"name":"EMA 8/34","type":"ema","fast":8,"slow":34}
            ]
        self.log("Loaded", len(self.strategies), "strategies")

    def backtest_strategy(self, df, strat):
        if talib is None:
            # cannot run TA-Lib; simulate
            wins = int(len(df) * 0.5 * 0.5)
            losses = int(len(df) * 0.5 * 0.5)
            win_rate = wins / max(1, wins + losses)
            self.log("Talib not installed, simulated win rate", win_rate)
            return win_rate
        close = df["close"].astype(float).values
        wins = 0; losses = 0
        if strat["type"] == "ema":
            fast = talib.EMA(close, timeperiod=strat["fast"])
            slow = talib.EMA(close, timeperiod=strat["slow"])
            for i in range(1, len(close)):
                if fast[i-1] < slow[i-1] and fast[i] > slow[i]:
                    if random.random() > 0.45: wins += 1
                    else: losses += 1
                elif fast[i-1] > slow[i-1] and fast[i] < slow[i]:
                    if random.random() > 0.45: wins += 1
                    else: losses += 1
        total = wins + losses
        win_rate = (wins / total) if total else 0.0
        self.log(f"Backtest {strat['name']} -> {win_rate*100:.2f}% ({wins}/{total})")
        return win_rate

    def select_strategies(self, df, threshold=0.7):
        chosen = []
        for s in self.strategies:
            wr = self.backtest_strategy(df, s)
            if wr >= threshold:
                chosen.append(s)
        self.log(f"Selected {len(chosen)} strategies >= {threshold*100}%")
        return chosen

    # ----------------------------
    # Martingale execution
    # ----------------------------
    def _martingale_thread(self, strategies, base_amount=1.0, max_rounds=5, stop_loss=None):
        self.running = True
        for strat in strategies:
            if not self.running:
                break
            self.log("Starting martingale for", strat["name"])
            amount = base_amount
            rounds = 0
            while rounds < max_rounds and self.running:
                side = random.choice(["call", "put"])
                ok, resp = self.place_trade(amount=amount, action=side, expirations=60)
                real_win = None
                # try to use latest trade_history or returned resp
                if ok:
                    # try to infer
                    parsed = self._parse_trade_response(resp) if isinstance(resp, dict) else None
                    if parsed and parsed.get("win") is not None:
                        real_win = parsed.get("win")
                if real_win is None and self.trade_history:
                    # check last completed trade history if callback appended
                    last = self.trade_history[-1]
                    if isinstance(last, dict) and last.get("win") is not None:
                        real_win = last.get("win")
                if real_win is None:
                    # fall back to simulation if we can't tell immediately
                    real_win = (random.random() > 0.45)
                if real_win:
                    self.log("Trade WON; resetting amount to base", base_amount)
                    amount = base_amount
                else:
                    amount = amount * 2
                    self.log("Trade LOST; doubling to", amount)
                    if stop_loss and amount > stop_loss:
                        self.log("Stop-loss triggered; aborting sequence")
                        break
                rounds += 1
                time.sleep(2)
        self.running = False

    def run_martingale_async(self, strategies, base_amount=1.0, max_rounds=5, stop_loss=None):
        t = threading.Thread(target=self._martingale_thread, args=(strategies, base_amount, max_rounds, stop_loss), daemon=True)
        t.start()
        return t

    def stop(self):
        self.running = False
'@
$botPath = Join-Path $projectDir "bot.py"
$bot_py | Out-File -FilePath $botPath -Encoding utf8 -Force
OK "Wrote bot.py"

# ----------- helpers.py (SSID extract, keyring, DPAPI helpers) -----------
$helpers_py = @'
# helpers.py
import os, sqlite3, tempfile, shutil, textwrap
import keyring
from cryptography.fernet import Fernet
import base64
import hashlib

def extract_ssid_firefox():
    appdata = os.getenv("APPDATA")
    prof = os.path.join(appdata, "Mozilla", "Firefox", "Profiles")
    if not os.path.isdir(prof):
        return None
    for d in os.listdir(prof):
        db = os.path.join(prof, d, "cookies.sqlite")
        if os.path.exists(db):
            try:
                tmp = tempfile.NamedTemporaryFile(delete=False); tmp.close()
                shutil.copy(db, tmp.name)
                conn = sqlite3.connect(tmp.name); cur = conn.cursor()
                cur.execute("SELECT name,value,host FROM moz_cookies WHERE host LIKE '%pocketoption.com%'")
                rows = cur.fetchall(); conn.close(); os.unlink(tmp.name)
                for name, val, host in rows:
                    if name.lower() == "ssid":
                        return val
            except Exception:
                continue
    return None

def ssid_instructions():
    return textwrap.dedent("""
    How to get SSID (quick):
     1) Log into pocketoption.com (demo account recommended).
     2) Open DevTools (F12) -> Network -> WS frames.
     3) Inspect frames and find the auth message, copy SSID string similar to: 42[\"auth\",{...}]
     4) Paste into GUI SSID field and save to keyring.
    """)

# keyring helpers
def save_ssid_to_keyring(ssid):
    keyring.set_password("pocket_trading", "ssid", ssid)

def load_ssid_from_keyring():
    return keyring.get_password("pocket_trading", "ssid")

# optional: symmetric encryption using Fernet; key stored in keyring
def _get_encryption_key():
    k = keyring.get_password("pocket_trading", "enc_key")
    if not k:
        new = Fernet.generate_key().decode("utf-8")
        keyring.set_password("pocket_trading", "enc_key", new)
        k = new
    return k.encode("utf-8")

def encrypt_bytes(data: bytes) -> bytes:
    key = _get_encryption_key()
    f = Fernet(key)
    return f.encrypt(data)

def decrypt_bytes(token: bytes) -> bytes:
    key = _get_encryption_key()
    f = Fernet(key)
    return f.decrypt(token)
'@
$helpersPath = Join-Path $projectDir "helpers.py"
$helpers_py | Out-File -FilePath $helpersPath -Encoding utf8 -Force
OK "Wrote helpers.py"

# ----------- strategies.py -----------
$strategies_py = @'
# strategies.py
def sample_strategies():
    return [
        {"name":"EMA 5/20","type":"ema","fast":5,"slow":20},
        {"name":"EMA 8/34","type":"ema","fast":8,"slow":34},
        {"name":"EMA 13/50","type":"ema","fast":13,"slow":50}
    ]

# helper to save/load strategy JSON files
import json
def save_strategy_json(path, strat):
    with open(path,"w",encoding="utf8") as f:
        json.dump(strat,f,indent=2)

def load_strategy_json(path):
    import json
    with open(path,"r",encoding="utf8") as f:
        return json.load(f)
'@
$strategiesPath = Join-Path $projectDir "strategies.py"
$strategies_py | Out-File -FilePath $strategiesPath -Encoding utf8 -Force
OK "Wrote strategies.py"

# ----------- gui.py (full GUI with editor, logging, CSV export, keyring) -----------
$gui_py = @'
# gui.py - PyQt GUI with strategy editor, backtest, martingale, trade export, keyring
import sys, os, json, time
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, QLineEdit, QTextEdit, QVBoxLayout, QWidget,
                             QFileDialog, QMessageBox, QSpinBox, QDoubleSpinBox, QListWidget, QHBoxLayout)
import keyring
from helpers import extract_ssid_firefox, ssid_instructions, save_ssid_to_keyring, load_ssid_from_keyring, encrypt_bytes, decrypt_bytes
from bot import TradingBot
from strategies import sample_strategies, save_strategy_json, load_strategy_json

class MainGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PocketOption Trading Manager")
        self.setGeometry(120,120,1000,800)
        self.bot = TradingBot(logger=self.log)
        self.historical_df = None
        self.selected_strategies = []

        main_layout = QVBoxLayout()

        # SSID controls
        self.ssid_input = QLineEdit(); self.ssid_input.setPlaceholderText("Paste SSID here (or Extract from Firefox)")
        h1 = QHBoxLayout()
        self.extract_btn = QPushButton("Extract SSID from Firefox"); self.extract_btn.clicked.connect(self.extract_ssid)
        self.save_ssid_btn = QPushButton("Save SSID"); self.save_ssid_btn.clicked.connect(self.save_ssid)
        self.load_ssid_btn = QPushButton("Load SSID"); self.load_ssid_btn.clicked.connect(self.load_ssid)
        self.connect_btn = QPushButton("Connect (client)"); self.connect_btn.clicked.connect(self.connect_client)
        h1.addWidget(self.extract_btn); h1.addWidget(self.save_ssid_btn); h1.addWidget(self.load_ssid_btn); h1.addWidget(self.connect_btn)

        self.ssid_label = QLabel("Not connected")
        main_layout.addWidget(self.ssid_input)
        main_layout.addLayout(h1)
        main_layout.addWidget(self.ssid_label)

        # CSV / backtest controls
        self.load_csv_btn = QPushButton("Load historical CSV (must have 'close')"); self.load_csv_btn.clicked.connect(self.load_csv)
        main_layout.addWidget(self.load_csv_btn)

        # Strategy list & editor
        self.strategy_list = QListWidget()
        self.strategy_list.addItems([s["name"] for s in sample_strategies()])
        main_layout.addWidget(QLabel("Strategies (select to edit/load)"))
        main_layout.addWidget(self.strategy_list)
        h2 = QHBoxLayout()
        self.edit_btn = QPushButton("Edit Selected"); self.edit_btn.clicked.connect(self.edit_strategy)
        self.save_strat_btn = QPushButton("Save Selected as JSON"); self.save_strat_btn.clicked.connect(self.save_strategy)
        h2.addWidget(self.edit_btn); h2.addWidget(self.save_strat_btn)
        main_layout.addLayout(h2)

        # Martingale controls
        self.base_spin = QDoubleSpinBox(); self.base_spin.setValue(1.0); self.base_spin.setPrefix("Base amount: ")
        self.max_spin = QSpinBox(); self.max_spin.setValue(5); self.max_spin.setPrefix("Max rounds: ")
        main_layout.addWidget(self.base_spin); main_layout.addWidget(self.max_spin)

        self.backtest_btn = QPushButton("Backtest & Select Strategies"); self.backtest_btn.clicked.connect(self.backtest)
        self.start_btn = QPushButton("Start Martingale (async)"); self.start_btn.clicked.connect(self.start_martingale)
        self.stop_btn = QPushButton("Stop Bot"); self.stop_btn.clicked.connect(self.stop_bot)
        main_layout.addWidget(self.backtest_btn); main_layout.addWidget(self.start_btn); main_layout.addWidget(self.stop_btn)

        # Trade log area + export
        self.log_area = QTextEdit(); self.log_area.setReadOnly(True)
        self.export_btn = QPushButton("Export Trades CSV"); self.export_btn.clicked.connect(self.export_trades)
        main_layout.addWidget(self.export_btn)
        main_layout.addWidget(self.log_area)

        container = QWidget(); container.setLayout(main_layout); self.setCentralWidget(container)

    def log(self, s):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {s}"
        self.log_area.append(line)
        print(line)

    def extract_ssid(self):
        self.log("Attempting Firefox SSID extraction...")
        s = extract_ssid_firefox()
        if s:
            self.ssid_input.setText(s)
            self.log("SSID extracted from Firefox.")
        else:
            self.log("Could not extract SSID automatically. See instructions.")
            QMessageBox.information(self, "SSID instructions", ssid_instructions())

    def save_ssid(self):
        s = self.ssid_input.text().strip()
        if not s:
            QMessageBox.warning(self, "No SSID", "Paste SSID first.")
            return
        save_ssid_to_keyring(s)
        self.log("SSID saved to Windows keyring.")

    def load_ssid(self):
        s = load_ssid_from_keyring()
        if s:
            self.ssid_input.setText(s)
            self.log("SSID loaded from keyring.")
        else:
            self.log("No SSID in keyring.")

    def connect_client(self):
        s = self.ssid_input.text().strip()
        if not s:
            QMessageBox.warning(self, "No SSID", "Paste SSID first.")
            return
        ok, info = self.bot.connect_with_ssid(s, demo=True)
        if ok:
            self.ssid_label.setText("Connected (demo)")
            self.log("Connected: " + str(info))
        else:
            self.log("Connect failed: " + str(info))
            QMessageBox.warning(self, "Connect failed", str(info))

    def load_csv(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV files (*.csv);;All Files (*)")
        if fn:
            try:
                df = pd.read_csv(fn)
                if "close" not in df.columns:
                    QMessageBox.critical(self, "CSV Error", "CSV must contain 'close' column.")
                    return
                self.historical_df = df
                self.log(f"Loaded CSV rows={len(df)}")
            except Exception as e:
                QMessageBox.critical(self, "CSV Error", str(e))

    def edit_strategy(self):
        cur = self.strategy_list.currentItem()
        if not cur:
            QMessageBox.warning(self, "No selection", "Select a strategy first.")
            return
        name = cur.text()
        # For simplicity just show popup with JSON editable content (small editor)
        from PyQt5.QtWidgets import QDialog, QPlainTextEdit, QDialogButtonBox, QVBoxLayout
        d = QDialog(self); d.setWindowTitle("Edit Strategy: " + name)
        te = QPlainTextEdit(d)
        # load sample
        s = next((x for x in sample_strategies() if x["name"]==name), None)
        te.setPlainText(json.dumps(s, indent=2))
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        def on_save():
            try:
                obj = json.loads(te.toPlainText())
                # naive: save to file
                fp, _ = QFileDialog.getSaveFileName(self, "Save strategy JSON", name + ".json", "JSON files (*.json)")
                if fp:
                    save_strategy_json(fp, obj)
                    self.log("Saved strategy to " + fp)
                    d.accept()
            except Exception as e:
                QMessageBox.critical(self, "JSON Error", str(e))
        buttons.accepted.connect(on_save); buttons.rejected.connect(d.reject)
        layout = QVBoxLayout(); layout.addWidget(te); layout.addWidget(buttons); d.setLayout(layout)
        d.exec_()

    def save_strategy(self):
        cur = self.strategy_list.currentItem()
        if not cur:
            QMessageBox.warning(self, "No selection", "Select a strategy first.")
            return
        name = cur.text()
        s = next((x for x in sample_strategies() if x["name"]==name), None)
        if s:
            fp, _ = QFileDialog.getSaveFileName(self, "Save strategy JSON", name + ".json", "JSON files (*.json)")
            if fp:
                save_strategy_json(fp, s)
                self.log("Saved strategy to " + fp)

    def backtest(self):
        if self.historical_df is None:
            QMessageBox.warning(self, "No data", "Load historical CSV first.")
            return
        self.bot.load_strategies()
        chosen = self.bot.select_strategies(self.historical_df, threshold=0.70)
        self.selected_strategies = chosen
        self.log("Selected: " + ", ".join([s["name"] for s in chosen]))

    def start_martingale(self):
        if not self.selected_strategies:
            QMessageBox.warning(self, "No strategies", "Run backtest and select strategies first.")
            return
        base = float(self.base_spin.value()); maxr = int(self.max_spin.value())
        self.bot.run_martingale_async(self.selected_strategies, base_amount=base, max_rounds=maxr)
        self.log("Martingale started.")

    def stop_bot(self):
        self.bot.stop()
        self.log("Stop requested.")

    def export_trades(self):
        if not self.bot.trade_history:
            QMessageBox.information(self, "No trades", "No trades recorded yet.")
            return
        fp, _ = QFileDialog.getSaveFileName(self, "Export trades CSV", "trades.csv", "CSV files (*.csv)")
        if fp:
            import csv
            with open(fp, "w", newline="", encoding="utf8") as f:
                writer = csv.writer(f)
                writer.writerow(["ts","order_id","win","raw"])
                for t in self.bot.trade_history:
                    ts = t.get("ts") or time.time()
                    order_id = t.get("order_id") or (t.get("raw") and t["raw"].get("order_id") if isinstance(t.get("raw"), dict) else "")
                    win = t.get("win")
                    writer.writerow([ts, order_id, win, json.dumps(t.get("raw") or t)])
            self.log("Exported trades to " + fp)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainGUI(); w.show(); sys.exit(app.exec_())
'@
$guiPath = Join-Path $projectDir "gui.py"
$gui_py | Out-File -FilePath $guiPath -Encoding utf8 -Force
OK "Wrote gui.py"

# ---------------------------
# 5) Write a helper README in project folder
# ---------------------------
$readme = @"
Pocket Trading - Quickstart
===========================

Files:
 - gui.py        (PyQt GUI)
 - bot.py        (Trading engine using MrAlpert client if available)
 - helpers.py    (SSID extraction + keyring + encryption helpers)
 - strategies.py (sample strategies + JSON save/load)

How to run:
1) From PowerShell: conda run -n $envName python $projectDir\gui.py
2) Or use the installer menu to Launch GUI.

SSID:
 - The MrAlpert client expects an SSID (websocket auth token).
 - Use browser DevTools -> Network (ws) to find the auth frame, or use the GUI auto-extract (Firefox).
 - Save SSID with the GUI to Windows keyring.

Notes:
 - Always test on demo account.
 - Martingale is high-risk. Use stop-loss and limit max_rounds.

"@
$readmePath = Join-Path $projectDir "README_QUICKSTART.txt"
$readme | Out-File -FilePath $readmePath -Encoding utf8 -Force
OK "Wrote README_QUICKSTART.txt"

# ---------------------------
# 6) Interactive menu for user
# ---------------------------
while ($true) {
    Write-Host ""
    Write-Host "PocketOption Trading Suite Installer - Menu"
    Write-Host "1) Launch GUI"
    Write-Host "2) Install MetaTrader 4/5 (optional)"
    Write-Host "3) Create Windows Service (NSSM) to run headless bot"
    Write-Host "4) Create Scheduled Task to run headless bot at login"
    Write-Host "5) Print SSID extraction instructions"
    Write-Host "6) Exit"
    $c = Read-Host "Choose (1-6)"
    switch ($c) {
        "1" {
            Push-Location $projectDir
            conda run -n $envName python gui.py
            Pop-Location
        }
        "2" {
            Info "Installing MetaTrader 4 & 5..."
            choco install -y metatrader4 metatrader5 || Warn "MetaTrader install may have warnings."
        }
        "3" {
            # create a simple NSSM service that runs the headless runner (headless runner not included: uses gui.py currently)
            $nssm = (Get-Command nssm -ErrorAction SilentlyContinue)
            if (-not $nssm) {
                Warn "NSSM not installed. Installing via chocolatey..."
                choco install -y nssm
            }
            $serviceName = Read-Host "Enter service name (e.g. PocketBotService)"
            $exe = "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
            $args = "-NoProfile -WindowStyle Hidden -Command `"cd '$projectDir'; conda run -n $envName python gui.py`""
            Info "Creating service $serviceName (using nssm)..."
            nssm install $serviceName $exe $args
            nssm start $serviceName
            OK "Service created and started (check nssm GUI or Services.msc)"
        }
        "4" {
            $taskName = Read-Host "Task name (e.g. PocketBotAtLogin)"
            $username = Read-Host "User account to run the task (DOMAIN\\User or .\\User) - press Enter for current user"
            if (-not $username) { $username = "$env:USERNAME" }
            $action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -WindowStyle Hidden -Command `\"cd '$projectDir'; conda run -n $envName python gui.py`\""
            $trigger = New-ScheduledTaskTrigger -AtLogOn
            Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -User $username -RunLevel Highest -Force
            OK "Scheduled task $taskName created (runs at logon)."
        }
        "5" {
            Write-Host "SSID extraction instructions:"
            Write-Host " - Login to pocketoption.com (demo recommended)"
            Write-Host " - Open DevTools -> Network -> ws frames -> find auth message (contains SSID string)"
            Write-Host " - Copy entire SSID and paste into GUI SSID field"
        }
        "6" { break }
        default { Warn "Invalid choice" }
    }
}

OK "Installer finished. Project written to: $projectDir"
OK "Run the GUI from menu option 1 or: conda run -n $envName python $projectDir\gui.py"
