#!/usr/bin/env python3
"""
GnuCash Database Access Patterns for ElizaOS-OpenCog Integration

Supports two storage backends:
  * SQLite  — GnuCash files saved with the SQLite backend (.gnucash opened
              with GnuCash ≥ 2.6 "Save as SQLite").
  * XML     — Classic GnuCash XML files (.gnucash or .xml, optionally
              gzip-compressed).  Parsed with the stdlib xml.etree module.

The public async API (get_accounts, get_transactions, …) is identical for both
backends so callers do not need to care which format is on disk.
"""

import gzip
import sqlite3
import asyncio
import logging
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, date, timedelta
from decimal import Decimal
from pathlib import Path

logger = logging.getLogger(__name__)

# GnuCash XML namespaces
_GNC_NS = {
    'gnc': 'http://www.gnucash.org/XML/gnc',
    'act': 'http://www.gnucash.org/XML/act',
    'trn': 'http://www.gnucash.org/XML/trn',
    'split': 'http://www.gnucash.org/XML/split',
    'ts': 'http://www.gnucash.org/XML/ts',
    'cmdty': 'http://www.gnucash.org/XML/cmdty',
}

class CurrencyConverter:
    """
    Currency conversion utility with exchange rate caching.
    Supports fetching rates from multiple sources and offline fallback.
    """
    
    # Default exchange rates (offline fallback, USD base)
    DEFAULT_RATES = {
        'USD': 1.0,
        'EUR': 0.92,
        'GBP': 0.79,
        'JPY': 149.50,
        'CAD': 1.36,
        'AUD': 1.53,
        'CHF': 0.88,
        'CNY': 7.24,
        'INR': 83.12,
        'MXN': 17.15,
        'BRL': 4.97,
        'KRW': 1320.0,
        'SGD': 1.34,
        'HKD': 7.82,
        'NOK': 10.65,
        'SEK': 10.42,
        'DKK': 6.87,
        'NZD': 1.64,
        'ZAR': 18.45,
        'RUB': 92.50,
    }
    
    def __init__(self, base_currency: str = 'USD'):
        self.base_currency = base_currency.upper()
        self.exchange_rates: Dict[str, float] = dict(self.DEFAULT_RATES)
        self.rates_updated: Optional[datetime] = None
        self.rate_sources: List[str] = ['exchangerate-api', 'fixer', 'openexchangerates']
        self._rate_cache: Dict[str, Dict[str, float]] = {}
        
    async def fetch_exchange_rates(self, 
                                   source: str = 'auto',
                                   api_key: Optional[str] = None) -> Dict[str, float]:
        """
        Fetch current exchange rates from external API.
        Falls back to cached rates if API is unavailable.
        
        Args:
            source: API source ('exchangerate-api', 'fixer', 'openexchangerates', 'auto')
            api_key: Optional API key for authenticated sources
            
        Returns:
            Dictionary of currency codes to exchange rates (base: USD)
        """
        import aiohttp
        
        apis = {
            'exchangerate-api': f'https://api.exchangerate-api.com/v4/latest/{self.base_currency}',
            'fixer': f'https://data.fixer.io/api/latest?access_key={api_key}&base={self.base_currency}' if api_key else None,
        }
        
        # Try sources in order
        sources_to_try = [source] if source != 'auto' else self.rate_sources
        
        for src in sources_to_try:
            url = apis.get(src)
            if not url:
                continue
                
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                        if response.status == 200:
                            data = await response.json()
                            rates = data.get('rates', data.get('conversion_rates', {}))
                            if rates:
                                self.exchange_rates = {self.base_currency: 1.0, **rates}
                                self.rates_updated = datetime.now()
                                logger.info(f"Exchange rates updated from {src}")
                                return self.exchange_rates
            except Exception as e:
                logger.debug(f"Failed to fetch rates from {src}: {e}")
                continue
        
        logger.warning("Using fallback exchange rates (offline mode)")
        return self.exchange_rates
    
    def convert(self, 
                amount: Decimal, 
                from_currency: str, 
                to_currency: str) -> Decimal:
        """
        Convert amount between currencies.
        
        Args:
            amount: Amount to convert
            from_currency: Source currency code (e.g., 'EUR')
            to_currency: Target currency code (e.g., 'USD')
            
        Returns:
            Converted amount in target currency
        """
        from_curr = from_currency.upper()
        to_curr = to_currency.upper()
        
        if from_curr == to_curr:
            return amount
        
        # Get rates relative to base currency
        from_rate = self.exchange_rates.get(from_curr, 1.0)
        to_rate = self.exchange_rates.get(to_curr, 1.0)
        
        # Convert: amount / from_rate * to_rate
        if from_rate == 0:
            from_rate = 1.0
        
        converted = Decimal(str(float(amount) / from_rate * to_rate))
        return converted.quantize(Decimal('0.01'))
    
    def get_rate(self, from_currency: str, to_currency: str) -> float:
        """Get exchange rate between two currencies."""
        from_rate = self.exchange_rates.get(from_currency.upper(), 1.0)
        to_rate = self.exchange_rates.get(to_currency.upper(), 1.0)
        
        if from_rate == 0:
            return 0.0
        return to_rate / from_rate
    
    def get_supported_currencies(self) -> List[str]:
        """Get list of supported currency codes."""
        return list(self.exchange_rates.keys())
    
    def add_custom_rate(self, currency: str, rate: float) -> None:
        """Add or update a custom exchange rate."""
        self.exchange_rates[currency.upper()] = rate
        
    def is_stale(self, max_age_hours: int = 24) -> bool:
        """Check if exchange rates are stale and need refresh."""
        if self.rates_updated is None:
            return True
        age = datetime.now() - self.rates_updated
        return age.total_seconds() > max_age_hours * 3600


class GnuCashDataAccess:
    """
    Core GnuCash database access layer with multi-currency support.

    Auto-detects the file format on :meth:`initialize`:
    * If the SQLite ``accounts`` table is present → SQLite backend.
    * Otherwise tries to parse as GnuCash XML (plain or gzip).
    * If the file does not exist at all → creates a minimal SQLite mock DB
      so unit tests work without a real GnuCash file.
    
    Multi-currency features:
    * Automatic currency detection from accounts/transactions
    * Exchange rate conversion with cached rates
    * Unified reporting in base currency
    """

    def __init__(self, database_path: str, base_currency: str = 'USD'):
        self.database_path = database_path
        self.connection = None
        self.initialized = False
        self._backend: str = "sqlite"  # "sqlite" | "xml"
        # In-memory tables populated when using the XML backend
        self._xml_accounts: List[Dict[str, Any]] = []
        self._xml_transactions: List[Dict[str, Any]] = []
        
        # Multi-currency support
        self.base_currency = base_currency.upper()
        self.currency_converter = CurrencyConverter(base_currency)
        self._currency_cache: Dict[str, str] = {}  # account_guid -> currency
        self._commodities: Dict[str, Dict[str, Any]] = {}  # guid -> commodity info
        
    async def initialize(self) -> bool:
        """Initialize GnuCash database connection.

        Detection order:
        1. File does not exist → create mock SQLite DB.
        2. File is a valid SQLite DB with GnuCash tables → SQLite backend.
        3. File is GnuCash XML (plain or gzip) → XML backend.
        4. Neither → log error and return False.
        """
        try:
            logger.info(f"Initializing GnuCash data access: {self.database_path}")
            path = Path(self.database_path)

            if not path.exists():
                logger.warning(f"GnuCash file not found: {self.database_path} — creating mock SQLite DB")
                await self._create_mock_database()

            # Try SQLite first
            if await self._try_sqlite_backend():
                self._backend = "sqlite"
                self.initialized = True
                logger.info("✅ GnuCash SQLite backend initialized")
                return True

            # Fall back to XML
            if await self._try_xml_backend():
                self._backend = "xml"
                self.initialized = True
                logger.info("✅ GnuCash XML backend initialized")
                return True

            logger.error("❌ Could not determine GnuCash file format")
            return False

        except Exception as e:
            logger.error(f"❌ Failed to initialize GnuCash data access: {e}")
            return False

    async def _try_sqlite_backend(self) -> bool:
        """Attempt to open as SQLite and verify the GnuCash schema."""
        try:
            conn = sqlite3.connect(self.database_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cursor.fetchall()}
            required = {'accounts', 'transactions', 'splits', 'commodities'}
            if required.issubset(tables):
                self.connection = conn
                return True
            conn.close()
            return False
        except Exception:
            return False

    async def _try_xml_backend(self) -> bool:
        """Attempt to parse as GnuCash XML (plain or gzip)."""
        try:
            path = Path(self.database_path)
            raw = path.read_bytes()
            # Detect gzip magic bytes
            if raw[:2] == b'\x1f\x8b':
                raw = gzip.decompress(raw)
            root = ET.fromstring(raw.decode('utf-8', errors='replace'))
            self._parse_xml_accounts(root)
            self._parse_xml_transactions(root)
            return True
        except Exception as exc:
            logger.debug(f"XML parse failed: {exc}")
            return False

    # ------------------------------------------------------------------
    # XML parsing helpers
    # ------------------------------------------------------------------

    def _xml_text(self, element: ET.Element, tag: str, ns_prefix: str = '') -> str:
        """Return text of first matching child element."""
        search = f"{{{_GNC_NS[ns_prefix]}}}{tag}" if ns_prefix else tag
        child = element.find(search)
        return child.text.strip() if child is not None and child.text else ''

    def _parse_xml_accounts(self, root: ET.Element) -> None:
        """Parse accounts from GnuCash XML into self._xml_accounts."""
        self._xml_accounts = []
        book = root.find(f"{{{_GNC_NS['gnc']}}}book")
        if book is None:
            return
        for acct in book.findall(f"{{{_GNC_NS['gnc']}}}account"):
            guid = self._xml_text(acct, 'id', 'act')
            name = self._xml_text(acct, 'name', 'act')
            acct_type = self._xml_text(acct, 'type', 'act')
            parent_el = acct.find(f"{{{_GNC_NS['act']}}}parent")
            parent_guid = parent_el.text.strip() if parent_el is not None and parent_el.text else None
            desc_el = acct.find(f"{{{_GNC_NS['act']}}}description")
            description = desc_el.text.strip() if desc_el is not None and desc_el.text else ''
            cmdty = acct.find(f"{{{_GNC_NS['act']}}}commodity")
            commodity_id = ''
            if cmdty is not None:
                id_el = cmdty.find(f"{{{_GNC_NS['cmdty']}}}id")
                if id_el is not None and id_el.text:
                    commodity_id = id_el.text.strip()
            self._xml_accounts.append({
                'guid': guid,
                'name': name,
                'account_type': acct_type,
                'commodity_guid': commodity_id,
                'parent_guid': parent_guid,
                'description': description,
            })

    def _parse_xml_transactions(self, root: ET.Element) -> None:
        """Parse transactions + splits from GnuCash XML."""
        self._xml_transactions = []
        book = root.find(f"{{{_GNC_NS['gnc']}}}book")
        if book is None:
            return
        # Build guid→name map for accounts
        acct_map = {a['guid']: a for a in self._xml_accounts}

        for txn in book.findall(f"{{{_GNC_NS['gnc']}}}transaction"):
            tx_guid = self._xml_text(txn, 'id', 'trn')
            desc = self._xml_text(txn, 'description', 'trn')
            date_posted_el = txn.find(
                f"{{{_GNC_NS['trn']}}}date-posted/{{{_GNC_NS['ts']}}}date"
            )
            post_date = ''
            if date_posted_el is not None and date_posted_el.text:
                raw_date = date_posted_el.text.strip()
                # GnuCash format: "2024-01-15 10:59:00 +0000"
                post_date = raw_date[:10]

            splits_el = txn.find(f"{{{_GNC_NS['trn']}}}splits")
            if splits_el is None:
                continue
            for split in splits_el.findall(f"{{{_GNC_NS['trn']}}}split"):
                split_guid = self._xml_text(split, 'id', 'split')
                acct_guid = self._xml_text(split, 'account', 'split')
                memo = self._xml_text(split, 'memo', 'split')
                value_str = self._xml_text(split, 'value', 'split')  # e.g. "-8550/100"
                try:
                    num, denom = value_str.split('/')
                    amount = Decimal(num) / Decimal(denom)
                except Exception:
                    amount = Decimal('0')

                acct = acct_map.get(acct_guid, {})
                self._xml_transactions.append({
                    'transaction_guid': tx_guid,
                    'description': desc,
                    'post_date': post_date,
                    'account_guid': acct_guid,
                    'account_name': acct.get('name', ''),
                    'account_type': acct.get('account_type', ''),
                    'memo': memo,
                    'amount': amount,
                    'enter_date': post_date,
                })
    
    async def _create_mock_database(self):
        """Create a mock GnuCash database for testing"""
        logger.info("Creating mock GnuCash database for integration testing...")
        
        # Create directory if needed
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Create database with basic GnuCash schema
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # Create accounts table
        cursor.execute('''
            CREATE TABLE accounts (
                guid TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                account_type TEXT NOT NULL,
                commodity_guid TEXT,
                commodity_scu INTEGER,
                non_std_scu INTEGER,
                parent_guid TEXT,
                code TEXT,
                description TEXT,
                hidden INTEGER,
                placeholder INTEGER
            )
        ''')
        
        # Create transactions table
        cursor.execute('''
            CREATE TABLE transactions (
                guid TEXT PRIMARY KEY,
                currency_guid TEXT,
                num TEXT,
                post_date DATE,
                enter_date TIMESTAMP,
                description TEXT
            )
        ''')
        
        # Create splits table
        cursor.execute('''
            CREATE TABLE splits (
                guid TEXT PRIMARY KEY,
                tx_guid TEXT,
                account_guid TEXT,
                memo TEXT,
                action TEXT,
                reconcile_state TEXT,
                reconcile_date TIMESTAMP,
                value_num INTEGER,
                value_denom INTEGER,
                quantity_num INTEGER,
                quantity_denom INTEGER,
                lot_guid TEXT
            )
        ''')
        
        # Create commodities table
        cursor.execute('''
            CREATE TABLE commodities (
                guid TEXT PRIMARY KEY,
                namespace TEXT,
                mnemonic TEXT,
                fullname TEXT,
                cusip TEXT,
                fraction INTEGER,
                quote_flag INTEGER,
                quote_source TEXT,
                quote_tz TEXT
            )
        ''')
        
        # Insert sample data
        await self._insert_sample_data(cursor)
        
        conn.commit()
        conn.close()
        logger.info("Mock GnuCash database created successfully")
    
    async def _insert_sample_data(self, cursor):
        """Insert sample financial data for testing"""
        # Insert USD commodity
        cursor.execute('''
            INSERT INTO commodities (guid, namespace, mnemonic, fullname, fraction)
            VALUES ('usd-guid', 'CURRENCY', 'USD', 'US Dollar', 100)
        ''')
        
        # Insert root account
        cursor.execute('''
            INSERT INTO accounts (guid, name, account_type, commodity_guid, parent_guid)
            VALUES ('root-guid', 'Root Account', 'ROOT', 'usd-guid', NULL)
        ''')
        
        # Insert asset accounts
        cursor.execute('''
            INSERT INTO accounts (guid, name, account_type, commodity_guid, parent_guid)
            VALUES ('checking-guid', 'Checking Account', 'BANK', 'usd-guid', 'root-guid')
        ''')
        
        cursor.execute('''
            INSERT INTO accounts (guid, name, account_type, commodity_guid, parent_guid)
            VALUES ('savings-guid', 'Savings Account', 'BANK', 'usd-guid', 'root-guid')
        ''')
        
        # Insert expense accounts
        cursor.execute('''
            INSERT INTO accounts (guid, name, account_type, commodity_guid, parent_guid)
            VALUES ('groceries-guid', 'Groceries', 'EXPENSE', 'usd-guid', 'root-guid')
        ''')
        
        cursor.execute('''
            INSERT INTO accounts (guid, name, account_type, commodity_guid, parent_guid)
            VALUES ('utilities-guid', 'Utilities', 'EXPENSE', 'usd-guid', 'root-guid')
        ''')
        
        # Insert sample transaction
        cursor.execute('''
            INSERT INTO transactions (guid, currency_guid, post_date, enter_date, description)
            VALUES ('tx1-guid', 'usd-guid', '2024-01-15', '2024-01-15 10:30:00', 'Grocery Shopping')
        ''')
        
        # Insert splits for the transaction
        cursor.execute('''
            INSERT INTO splits (guid, tx_guid, account_guid, memo, value_num, value_denom, quantity_num, quantity_denom)
            VALUES ('split1-guid', 'tx1-guid', 'checking-guid', 'Payment for groceries', -8550, 100, -8550, 100)
        ''')
        
        cursor.execute('''
            INSERT INTO splits (guid, tx_guid, account_guid, memo, value_num, value_denom, quantity_num, quantity_denom)
            VALUES ('split2-guid', 'tx1-guid', 'groceries-guid', 'Grocery purchase', 8550, 100, 8550, 100)
        ''')
    
    async def get_accounts(self, account_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get accounts, optionally filtered by type."""
        if not self.initialized:
            raise RuntimeError("Database not initialized")

        if self._backend == "xml":
            accounts = self._xml_accounts
            if account_type:
                accounts = [a for a in accounts if a['account_type'] == account_type]
            return sorted(accounts, key=lambda a: a['name'])

        # SQLite backend
        cursor = self.connection.cursor()
        if account_type:
            cursor.execute(
                "SELECT * FROM accounts WHERE account_type = ? ORDER BY name",
                (account_type,)
            )
        else:
            cursor.execute("SELECT * FROM accounts ORDER BY name")

        accounts = []
        for row in cursor.fetchall():
            accounts.append({
                'guid': row['guid'],
                'name': row['name'],
                'account_type': row['account_type'],
                'commodity_guid': row['commodity_guid'],
                'parent_guid': row['parent_guid'],
                'description': row['description']
            })
        return accounts
    
    async def get_account_balance(self, account_guid: str) -> Decimal:
        """Get current balance for an account"""
        if not self.initialized:
            raise RuntimeError("Database not initialized")
        
        cursor = self.connection.cursor()
        
        cursor.execute('''
            SELECT SUM(CAST(value_num AS REAL) / value_denom) as balance
            FROM splits 
            WHERE account_guid = ?
        ''', (account_guid,))
        
        result = cursor.fetchone()
        balance = result['balance'] if result['balance'] else 0.0
        
        return Decimal(str(balance))
    
    async def get_transactions(self,
                             account_guid: Optional[str] = None,
                             start_date: Optional[date] = None,
                             end_date: Optional[date] = None,
                             limit: int = 100) -> List[Dict[str, Any]]:
        """Get transactions with optional filtering."""
        if not self.initialized:
            raise RuntimeError("Database not initialized")

        if self._backend == "xml":
            rows = self._xml_transactions
            if account_guid:
                rows = [r for r in rows if r['account_guid'] == account_guid]
            if start_date:
                rows = [r for r in rows if r['post_date'] >= start_date.isoformat()]
            if end_date:
                rows = [r for r in rows if r['post_date'] <= end_date.isoformat()]
            rows = sorted(rows, key=lambda r: r['post_date'], reverse=True)[:limit]
            return rows

        # SQLite backend
        cursor = self.connection.cursor()
        query = '''
            SELECT t.*, s.account_guid, s.memo, s.value_num, s.value_denom,
                   a.name as account_name, a.account_type
            FROM transactions t
            JOIN splits s ON t.guid = s.tx_guid
            JOIN accounts a ON s.account_guid = a.guid
            WHERE 1=1
        '''
        params = []
        if account_guid:
            query += " AND s.account_guid = ?"
            params.append(account_guid)
        if start_date:
            query += " AND t.post_date >= ?"
            params.append(start_date.isoformat())
        if end_date:
            query += " AND t.post_date <= ?"
            params.append(end_date.isoformat())
        query += " ORDER BY t.post_date DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        transactions = []
        for row in cursor.fetchall():
            transactions.append({
                'transaction_guid': row['guid'],
                'description': row['description'],
                'post_date': row['post_date'],
                'account_guid': row['account_guid'],
                'account_name': row['account_name'],
                'account_type': row['account_type'],
                'memo': row['memo'],
                'amount': Decimal(row['value_num']) / Decimal(row['value_denom']),
                'enter_date': row['enter_date']
            })
        return transactions
    
    async def get_spending_by_category(self,
                                     start_date: date,
                                     end_date: date) -> Dict[str, Decimal]:
        """Get spending totals by expense category."""
        if not self.initialized:
            raise RuntimeError("Database not initialized")

        if self._backend == "xml":
            spending: Dict[str, Decimal] = {}
            for row in self._xml_transactions:
                if row['account_type'] != 'EXPENSE':
                    continue
                if row['post_date'] < start_date.isoformat() or row['post_date'] > end_date.isoformat():
                    continue
                if row['amount'] > 0:
                    name = row['account_name']
                    spending[name] = spending.get(name, Decimal('0')) + row['amount']
            return dict(sorted(spending.items(), key=lambda x: x[1], reverse=True))

        cursor = self.connection.cursor()
        cursor.execute('''
            SELECT a.name as category,
                   SUM(CAST(s.value_num AS REAL) / s.value_denom) as total
            FROM transactions t
            JOIN splits s ON t.guid = s.tx_guid
            JOIN accounts a ON s.account_guid = a.guid
            WHERE a.account_type = 'EXPENSE'
            AND t.post_date >= ?
            AND t.post_date <= ?
            AND s.value_num > 0
            GROUP BY a.name
            ORDER BY total DESC
        ''', (start_date.isoformat(), end_date.isoformat()))

        spending = {}
        for row in cursor.fetchall():
            category = row['category']
            total = Decimal(str(row['total'])) if row['total'] else Decimal('0')
            spending[category] = total
        return spending
    
    async def get_income_by_category(self,
                                   start_date: date,
                                   end_date: date) -> Dict[str, Decimal]:
        """Get income totals by income category"""
        if not self.initialized:
            raise RuntimeError("Database not initialized")
        
        cursor = self.connection.cursor()
        
        cursor.execute('''
            SELECT a.name as category,
                   SUM(CAST(s.value_num AS REAL) / s.value_denom) as total
            FROM transactions t
            JOIN splits s ON t.guid = s.tx_guid  
            JOIN accounts a ON s.account_guid = a.guid
            WHERE a.account_type = 'INCOME'
            AND t.post_date >= ?
            AND t.post_date <= ?
            AND s.value_num < 0
            GROUP BY a.name
            ORDER BY total DESC
        ''', (start_date.isoformat(), end_date.isoformat()))
        
        income = {}
        for row in cursor.fetchall():
            category = row['category']
            total = abs(Decimal(str(row['total']))) if row['total'] else Decimal('0')
            income[category] = total
        
        return income
    
    async def search_transactions(self, search_term: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search transactions by description or memo."""
        if not self.initialized:
            raise RuntimeError("Database not initialized")

        if self._backend == "xml":
            term = search_term.lower()
            results = [
                r for r in self._xml_transactions
                if term in r['description'].lower() or term in r['memo'].lower()
            ]
            return results[:limit]

        cursor = self.connection.cursor()
        search_pattern = f"%{search_term}%"
        cursor.execute('''
            SELECT t.*, s.account_guid, s.memo, s.value_num, s.value_denom,
                   a.name as account_name, a.account_type
            FROM transactions t
            JOIN splits s ON t.guid = s.tx_guid
            JOIN accounts a ON s.account_guid = a.guid
            WHERE (t.description LIKE ? OR s.memo LIKE ?)
            ORDER BY t.post_date DESC
            LIMIT ?
        ''', (search_pattern, search_pattern, limit))

        results = []
        for row in cursor.fetchall():
            results.append({
                'transaction_guid': row['guid'],
                'description': row['description'],
                'post_date': row['post_date'],
                'account_name': row['account_name'],
                'memo': row['memo'],
                'amount': Decimal(row['value_num']) / Decimal(row['value_denom'])
            })
        return results
    
    # -------------------------------------------------------------------------
    # Multi-currency support methods
    # -------------------------------------------------------------------------
    
    async def load_commodities(self) -> Dict[str, Dict[str, Any]]:
        """Load all commodities (currencies and securities) from the database."""
        if not self.initialized:
            raise RuntimeError("Database not initialized")
        
        if self._backend == "xml":
            # For XML backend, commodities are embedded in accounts
            for acct in self._xml_accounts:
                commodity_id = acct.get('commodity_guid', 'USD')
                if commodity_id and commodity_id not in self._commodities:
                    self._commodities[commodity_id] = {
                        'guid': commodity_id,
                        'mnemonic': commodity_id,
                        'namespace': 'CURRENCY',
                        'fullname': commodity_id,
                        'fraction': 100
                    }
            return self._commodities
        
        cursor = self.connection.cursor()
        cursor.execute('''
            SELECT guid, namespace, mnemonic, fullname, fraction
            FROM commodities
        ''')
        
        for row in cursor.fetchall():
            self._commodities[row['guid']] = {
                'guid': row['guid'],
                'mnemonic': row['mnemonic'],
                'namespace': row['namespace'],
                'fullname': row['fullname'],
                'fraction': row['fraction']
            }
        
        return self._commodities
    
    async def get_account_currency(self, account_guid: str) -> str:
        """Get the currency code for a specific account."""
        if account_guid in self._currency_cache:
            return self._currency_cache[account_guid]
        
        if not self._commodities:
            await self.load_commodities()
        
        if self._backend == "xml":
            for acct in self._xml_accounts:
                if acct['guid'] == account_guid:
                    commodity_id = acct.get('commodity_guid', self.base_currency)
                    self._currency_cache[account_guid] = commodity_id
                    return commodity_id
            return self.base_currency
        
        cursor = self.connection.cursor()
        cursor.execute('''
            SELECT c.mnemonic
            FROM accounts a
            JOIN commodities c ON a.commodity_guid = c.guid
            WHERE a.guid = ?
        ''', (account_guid,))
        
        row = cursor.fetchone()
        currency = row['mnemonic'] if row else self.base_currency
        self._currency_cache[account_guid] = currency
        return currency
    
    async def get_transactions_with_currency(self,
                                            start_date: date,
                                            end_date: date,
                                            convert_to_base: bool = True) -> List[Dict[str, Any]]:
        """
        Get transactions with currency information.
        
        Args:
            start_date: Start date for transaction range
            end_date: End date for transaction range
            convert_to_base: If True, convert all amounts to base currency
            
        Returns:
            List of transactions with currency and optional converted amount
        """
        if not self.initialized:
            raise RuntimeError("Database not initialized")
        
        # Ensure commodities are loaded
        if not self._commodities:
            await self.load_commodities()
        
        if self._backend == "xml":
            transactions = []
            for t in self._xml_transactions:
                post_date = t.get('post_date', '')
                if start_date.isoformat() <= post_date <= end_date.isoformat():
                    currency = await self.get_account_currency(t.get('account_guid', ''))
                    amount = t.get('amount', Decimal('0'))
                    
                    tx_data = {
                        **t,
                        'currency': currency,
                        'original_amount': amount,
                    }
                    
                    if convert_to_base and currency != self.base_currency:
                        tx_data['amount'] = self.currency_converter.convert(
                            amount, currency, self.base_currency
                        )
                        tx_data['converted_to'] = self.base_currency
                    
                    transactions.append(tx_data)
            return transactions
        
        cursor = self.connection.cursor()
        cursor.execute('''
            SELECT t.guid as transaction_guid, t.description, t.post_date,
                   s.account_guid, s.memo, s.value_num, s.value_denom,
                   a.name as account_name, a.account_type, a.commodity_guid,
                   c.mnemonic as currency
            FROM transactions t
            JOIN splits s ON t.guid = s.tx_guid
            JOIN accounts a ON s.account_guid = a.guid
            LEFT JOIN commodities c ON a.commodity_guid = c.guid
            WHERE t.post_date >= ? AND t.post_date <= ?
            ORDER BY t.post_date DESC
        ''', (start_date.isoformat(), end_date.isoformat()))
        
        transactions = []
        for row in cursor.fetchall():
            currency = row['currency'] or self.base_currency
            amount = Decimal(row['value_num']) / Decimal(row['value_denom'])
            
            tx_data = {
                'transaction_guid': row['transaction_guid'],
                'description': row['description'],
                'post_date': row['post_date'],
                'account_guid': row['account_guid'],
                'account_name': row['account_name'],
                'account_type': row['account_type'],
                'memo': row['memo'],
                'currency': currency,
                'original_amount': amount,
                'amount': amount
            }
            
            if convert_to_base and currency != self.base_currency:
                tx_data['amount'] = self.currency_converter.convert(
                    amount, currency, self.base_currency
                )
                tx_data['converted_to'] = self.base_currency
            
            transactions.append(tx_data)
        
        return transactions
    
    async def get_multi_currency_balance(self,
                                         account_guid: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Get account balances organized by currency.
        
        Args:
            account_guid: Optional specific account, or all accounts if None
            
        Returns:
            Dictionary with currency codes as keys, containing balance info
        """
        if not self.initialized:
            raise RuntimeError("Database not initialized")
        
        if not self._commodities:
            await self.load_commodities()
        
        balances_by_currency: Dict[str, Dict[str, Any]] = {}
        
        if self._backend == "xml":
            for t in self._xml_transactions:
                if account_guid and t.get('account_guid') != account_guid:
                    continue
                
                currency = await self.get_account_currency(t.get('account_guid', ''))
                amount = t.get('amount', Decimal('0'))
                
                if currency not in balances_by_currency:
                    balances_by_currency[currency] = {
                        'currency': currency,
                        'total_balance': Decimal('0'),
                        'accounts': {},
                        'base_currency_equivalent': Decimal('0')
                    }
                
                acct_name = t.get('account_name', 'Unknown')
                if acct_name not in balances_by_currency[currency]['accounts']:
                    balances_by_currency[currency]['accounts'][acct_name] = Decimal('0')
                
                balances_by_currency[currency]['accounts'][acct_name] += amount
                balances_by_currency[currency]['total_balance'] += amount
        else:
            cursor = self.connection.cursor()
            
            query = '''
                SELECT a.guid, a.name, a.account_type,
                       COALESCE(c.mnemonic, 'USD') as currency,
                       SUM(CAST(s.value_num AS REAL) / s.value_denom) as balance
                FROM accounts a
                LEFT JOIN splits s ON a.guid = s.account_guid
                LEFT JOIN commodities c ON a.commodity_guid = c.guid
            '''
            
            if account_guid:
                query += ' WHERE a.guid = ?'
                cursor.execute(query + ' GROUP BY a.guid', (account_guid,))
            else:
                cursor.execute(query + ' GROUP BY a.guid')
            
            for row in cursor.fetchall():
                currency = row['currency'] or self.base_currency
                balance = Decimal(str(row['balance'] or 0))
                
                if currency not in balances_by_currency:
                    balances_by_currency[currency] = {
                        'currency': currency,
                        'total_balance': Decimal('0'),
                        'accounts': {},
                        'base_currency_equivalent': Decimal('0')
                    }
                
                balances_by_currency[currency]['accounts'][row['name']] = balance
                balances_by_currency[currency]['total_balance'] += balance
        
        # Calculate base currency equivalents
        for currency, data in balances_by_currency.items():
            if currency == self.base_currency:
                data['base_currency_equivalent'] = data['total_balance']
            else:
                data['base_currency_equivalent'] = self.currency_converter.convert(
                    data['total_balance'], currency, self.base_currency
                )
        
        return balances_by_currency
    
    async def get_spending_by_currency(self,
                                       start_date: date,
                                       end_date: date) -> Dict[str, Dict[str, Decimal]]:
        """
        Get spending breakdown by currency and category.
        
        Returns:
            Dictionary with currency codes as keys, containing category spending
        """
        if not self.initialized:
            raise RuntimeError("Database not initialized")
        
        transactions = await self.get_transactions_with_currency(
            start_date, end_date, convert_to_base=False
        )
        
        spending_by_currency: Dict[str, Dict[str, Decimal]] = {}
        
        for tx in transactions:
            if tx.get('account_type') != 'EXPENSE':
                continue
            
            currency = tx.get('currency', self.base_currency)
            category = tx.get('account_name', 'Uncategorized')
            amount = abs(tx.get('original_amount', Decimal('0')))
            
            if currency not in spending_by_currency:
                spending_by_currency[currency] = {}
            
            if category not in spending_by_currency[currency]:
                spending_by_currency[currency][category] = Decimal('0')
            
            spending_by_currency[currency][category] += amount
        
        return spending_by_currency
    
    async def refresh_exchange_rates(self, api_key: Optional[str] = None) -> Dict[str, float]:
        """Refresh exchange rates from external API."""
        return await self.currency_converter.fetch_exchange_rates(api_key=api_key)
    
    def convert_amount(self, 
                      amount: Decimal, 
                      from_currency: str, 
                      to_currency: Optional[str] = None) -> Decimal:
        """
        Convert an amount between currencies.
        
        Args:
            amount: Amount to convert
            from_currency: Source currency code
            to_currency: Target currency (defaults to base_currency)
            
        Returns:
            Converted amount
        """
        target = to_currency or self.base_currency
        return self.currency_converter.convert(amount, from_currency, target)
    
    async def get_currency_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all currencies used in the database.
        
        Returns:
            Summary including currency list, default currency, and exchange rates
        """
        if not self._commodities:
            await self.load_commodities()
        
        currencies = [
            c for c in self._commodities.values()
            if c.get('namespace') == 'CURRENCY'
        ]
        
        balances = await self.get_multi_currency_balance()
        
        return {
            'base_currency': self.base_currency,
            'currencies_in_use': list(balances.keys()),
            'available_currencies': [c['mnemonic'] for c in currencies],
            'exchange_rates': dict(self.currency_converter.exchange_rates),
            'rates_updated': self.currency_converter.rates_updated.isoformat() if self.currency_converter.rates_updated else None,
            'total_in_base_currency': sum(
                data['base_currency_equivalent'] for data in balances.values()
            )
        }
    
    async def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
            self.initialized = False
            logger.info("GnuCash database connection closed")


class FinancialPatternAnalyzer:
    """
    Analyzer for detecting patterns in financial data
    Provides foundation for cognitive analysis integration
    """
    
    def __init__(self, data_access: GnuCashDataAccess):
        self.data_access = data_access
    
    async def analyze_spending_trends(self, 
                                    months: int = 6) -> Dict[str, Any]:
        """Analyze spending trends over time"""
        end_date = date.today()
        start_date = date(end_date.year, end_date.month - months, 1)
        
        # Get spending data
        spending = await self.data_access.get_spending_by_category(start_date, end_date)
        
        # Calculate trends
        total_spending = sum(spending.values())
        category_percentages = {}
        
        for category, amount in spending.items():
            percentage = (amount / total_spending * 100) if total_spending > 0 else 0
            category_percentages[category] = float(percentage)
        
        # Identify top categories
        top_categories = sorted(spending.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'period': f"{start_date} to {end_date}",
            'total_spending': float(total_spending),
            'category_breakdown': {k: float(v) for k, v in spending.items()},
            'category_percentages': category_percentages,
            'top_categories': [(cat, float(amt)) for cat, amt in top_categories],
            'analysis_insights': await self._generate_spending_insights(spending, total_spending)
        }
    
    async def _generate_spending_insights(self, 
                                        spending: Dict[str, Decimal],
                                        total: Decimal) -> List[str]:
        """Generate insights from spending data"""
        insights = []
        
        if total > Decimal('1000'):
            insights.append("Significant spending activity detected")
        
        # Find largest category
        if spending:
            largest_category = max(spending.items(), key=lambda x: x[1])
            percentage = (largest_category[1] / total * 100) if total > 0 else 0
            
            if percentage > 40:
                insights.append(f"High concentration of spending in {largest_category[0]} ({percentage:.1f}%)")
            
            if 'Groceries' in spending and spending['Groceries'] > total * Decimal('0.15'):
                insights.append("Grocery spending represents significant portion of expenses")
        
        if len(spending) > 5:
            insights.append("Diverse spending across multiple categories")
        elif len(spending) <= 2:
            insights.append("Concentrated spending in few categories")
        
        return insights
    
    async def detect_anomalies(self, 
                             account_guid: str,
                             days: int = 30) -> List[Dict[str, Any]]:
        """Detect unusual transactions or patterns"""
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        
        transactions = await self.data_access.get_transactions(
            account_guid=account_guid,
            start_date=start_date,
            end_date=end_date
        )
        
        if not transactions:
            return []
        
        # Calculate statistics
        amounts = [abs(float(tx['amount'])) for tx in transactions]
        mean_amount = sum(amounts) / len(amounts)
        
        # Simple anomaly detection - amounts > 2 standard deviations from mean
        import statistics
        std_dev = statistics.stdev(amounts) if len(amounts) > 1 else 0
        threshold = mean_amount + (2 * std_dev)
        
        anomalies = []
        for tx in transactions:
            amount = abs(float(tx['amount']))
            if amount > threshold:
                anomalies.append({
                    'transaction': tx,
                    'anomaly_type': 'high_amount',
                    'amount': amount,
                    'threshold': threshold,
                    'deviation': amount - mean_amount
                })
        
        return anomalies