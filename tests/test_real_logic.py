"""
Unit tests for the real (non-mock) logic added in the Priority 1–2 implementation.

These tests run offline without any external services.  They verify:
- GnuCash SQLite backend works correctly
- GnuCash XML backend parses a minimal in-memory document
- AI categorizer pattern matching and ML training
- Cognitive anomaly detection (Z-score + IQR)
- NL query interface intent parsing
"""

import asyncio
import os
import sys
import tempfile
import textwrap
import sqlite3
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(coro):
    """Convenience wrapper to run async code in tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


def make_sqlite_db(path: str) -> None:
    """Create a minimal GnuCash SQLite database at *path*."""
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute(
        "CREATE TABLE commodities (guid TEXT PRIMARY KEY, namespace TEXT, mnemonic TEXT, "
        "fullname TEXT, cusip TEXT, fraction INTEGER, quote_flag INTEGER, "
        "quote_source TEXT, quote_tz TEXT)"
    )
    c.execute(
        "CREATE TABLE accounts (guid TEXT PRIMARY KEY, name TEXT, account_type TEXT, "
        "commodity_guid TEXT, commodity_scu INTEGER, non_std_scu INTEGER, "
        "parent_guid TEXT, code TEXT, description TEXT, hidden INTEGER, placeholder INTEGER)"
    )
    c.execute(
        "CREATE TABLE transactions (guid TEXT PRIMARY KEY, currency_guid TEXT, num TEXT, "
        "post_date DATE, enter_date TIMESTAMP, description TEXT)"
    )
    c.execute(
        "CREATE TABLE splits (guid TEXT PRIMARY KEY, tx_guid TEXT, account_guid TEXT, "
        "memo TEXT, action TEXT, reconcile_state TEXT, reconcile_date TIMESTAMP, "
        "value_num INTEGER, value_denom INTEGER, quantity_num INTEGER, "
        "quantity_denom INTEGER, lot_guid TEXT)"
    )
    c.execute("INSERT INTO commodities VALUES ('usd','CURRENCY','USD','US Dollar','',100,0,'','')")
    c.execute(
        "INSERT INTO accounts VALUES "
        "('root','Root','ROOT','usd',100,0,NULL,'','',0,0)"
    )
    c.execute(
        "INSERT INTO accounts VALUES "
        "('chk','Checking','BANK','usd',100,0,'root','','',0,0)"
    )
    c.execute(
        "INSERT INTO accounts VALUES "
        "('groc','Groceries','EXPENSE','usd',100,0,'root','','',0,0)"
    )
    c.execute(
        "INSERT INTO transactions VALUES "
        "('tx1','usd','','2024-01-10','2024-01-10 09:00:00','Supermarket')"
    )
    c.execute(
        "INSERT INTO splits VALUES "
        "('s1','tx1','chk','','','n',NULL,-5000,100,-5000,100,NULL)"
    )
    c.execute(
        "INSERT INTO splits VALUES "
        "('s2','tx1','groc','','','n',NULL,5000,100,5000,100,NULL)"
    )
    conn.commit()
    conn.close()


MINIMAL_XML = textwrap.dedent("""\
<?xml version="1.0" encoding="utf-8" ?>
<gnc-v2 xmlns:gnc="http://www.gnucash.org/XML/gnc"
        xmlns:act="http://www.gnucash.org/XML/act"
        xmlns:trn="http://www.gnucash.org/XML/trn"
        xmlns:split="http://www.gnucash.org/XML/split"
        xmlns:ts="http://www.gnucash.org/XML/ts"
        xmlns:cmdty="http://www.gnucash.org/XML/cmdty">
  <gnc:book>
    <gnc:account version="2.0.0">
      <act:name>Checking</act:name>
      <act:id type="guid">chk-xml</act:id>
      <act:type>BANK</act:type>
      <act:commodity><cmdty:id>USD</cmdty:id></act:commodity>
    </gnc:account>
    <gnc:account version="2.0.0">
      <act:name>Dining</act:name>
      <act:id type="guid">din-xml</act:id>
      <act:type>EXPENSE</act:type>
      <act:commodity><cmdty:id>USD</cmdty:id></act:commodity>
    </gnc:account>
    <gnc:transaction version="2.0.0">
      <trn:id type="guid">txxml1</trn:id>
      <trn:description>Restaurant visit</trn:description>
      <trn:date-posted><ts:date>2024-03-05 00:00:00 +0000</ts:date></trn:date-posted>
      <trn:splits>
        <trn:split>
          <split:id type="guid">sp1</split:id>
          <split:account type="guid">chk-xml</split:account>
          <split:memo></split:memo>
          <split:value>-3500/100</split:value>
        </trn:split>
        <trn:split>
          <split:id type="guid">sp2</split:id>
          <split:account type="guid">din-xml</split:account>
          <split:memo></split:memo>
          <split:value>3500/100</split:value>
        </trn:split>
      </trn:splits>
    </gnc:transaction>
  </gnc:book>
</gnc-v2>
""")


# ---------------------------------------------------------------------------
# GnuCash SQLite backend tests
# ---------------------------------------------------------------------------

class TestGnuCashSQLiteBackend:
    def setup_method(self):
        from core.gnucash_access import GnuCashDataAccess
        self._tmp = tempfile.NamedTemporaryFile(suffix=".gnucash", delete=False)
        self._tmp.close()
        make_sqlite_db(self._tmp.name)
        self.db = GnuCashDataAccess(self._tmp.name)

    def teardown_method(self):
        run(self.db.close())
        os.unlink(self._tmp.name)

    def test_initialize_sqlite(self):
        ok = run(self.db.initialize())
        assert ok, "SQLite backend should initialize successfully"
        assert self.db._backend == "sqlite"

    def test_get_accounts(self):
        run(self.db.initialize())
        accounts = run(self.db.get_accounts())
        names = [a["name"] for a in accounts]
        assert "Checking" in names
        assert "Groceries" in names

    def test_get_accounts_filtered(self):
        run(self.db.initialize())
        expense_accounts = run(self.db.get_accounts(account_type="EXPENSE"))
        assert all(a["account_type"] == "EXPENSE" for a in expense_accounts)

    def test_get_transactions(self):
        run(self.db.initialize())
        txns = run(self.db.get_transactions())
        assert len(txns) > 0
        assert txns[0]["description"] == "Supermarket"

    def test_search_transactions(self):
        run(self.db.initialize())
        results = run(self.db.search_transactions("super"))
        assert len(results) > 0

    def test_get_account_balance(self):
        run(self.db.initialize())
        accounts = run(self.db.get_accounts(account_type="EXPENSE"))
        assert len(accounts) > 0
        bal = run(self.db.get_account_balance(accounts[0]["guid"]))
        assert float(bal) == 50.0  # 5000/100


# ---------------------------------------------------------------------------
# GnuCash XML backend tests
# ---------------------------------------------------------------------------

class TestGnuCashXMLBackend:
    def setup_method(self):
        from core.gnucash_access import GnuCashDataAccess
        self._tmp = tempfile.NamedTemporaryFile(
            suffix=".gnucash", delete=False, mode="w", encoding="utf-8"
        )
        self._tmp.write(MINIMAL_XML)
        self._tmp.close()
        self.db = GnuCashDataAccess(self._tmp.name)

    def teardown_method(self):
        run(self.db.close())
        os.unlink(self._tmp.name)

    def test_initialize_xml(self):
        ok = run(self.db.initialize())
        assert ok, "XML backend should initialize successfully"
        assert self.db._backend == "xml"

    def test_get_accounts_xml(self):
        run(self.db.initialize())
        accounts = run(self.db.get_accounts())
        names = [a["name"] for a in accounts]
        assert "Checking" in names
        assert "Dining" in names

    def test_get_transactions_xml(self):
        run(self.db.initialize())
        txns = run(self.db.get_transactions())
        assert len(txns) == 2  # two splits for one transaction
        descs = {t["description"] for t in txns}
        assert "Restaurant visit" in descs

    def test_get_spending_by_category_xml(self):
        run(self.db.initialize())
        from datetime import date
        spending = run(self.db.get_spending_by_category(
            date(2024, 1, 1), date(2024, 12, 31)
        ))
        assert "Dining" in spending
        assert float(spending["Dining"]) == 35.0  # 3500/100


# ---------------------------------------------------------------------------
# AI categorizer tests
# ---------------------------------------------------------------------------

class TestAITransactionCategorizer:
    def setup_method(self):
        from financial.ai_categorization import AITransactionCategorizer
        self.cat = AITransactionCategorizer()

    def test_pattern_groceries(self):
        result = run(self.cat.categorize_transaction(
            {"description": "whole foods market", "amount": 55.0}
        ))
        assert result.category == "groceries"
        assert result.confidence > 0.5

    def test_pattern_dining(self):
        result = run(self.cat.categorize_transaction(
            {"description": "starbucks coffee", "amount": 6.0}
        ))
        assert result.category == "dining"

    def test_income_heuristic(self):
        # Use a description that doesn't match any regex pattern so the
        # amount-based heuristic (negative amount → income) fires.
        result = run(self.cat.categorize_transaction(
            {"description": "xyzzy payment received", "amount": -2000.0}
        ))
        assert result.category == "income"

    def test_uncategorized(self):
        result = run(self.cat.categorize_transaction(
            {"description": "zzzzunknown", "amount": 10.0}
        ))
        assert result.category == "uncategorized"

    def test_ml_training_from_labelled_data(self):
        """Train ML model from labelled data and check it produces predictions."""
        try:
            from sklearn.linear_model import LogisticRegression  # noqa
        except ImportError:
            pytest.skip("scikit-learn not installed")

        labelled = []
        for _ in range(10):
            labelled.append(("supermarket grocery store food", "groceries"))
            labelled.append(("restaurant dinner pizza burger", "dining"))
            labelled.append(("electric bill power utility", "utilities"))
        trained = self.cat.train_from_labelled_data(labelled)
        assert trained, "Model should be trained with sufficient data"
        assert self.cat._ml_trained

        result = run(self.cat.categorize_transaction(
            {"description": "pizza restaurant visit", "amount": 20.0}
        ))
        assert result.category in {"dining", "groceries", "utilities"}

    def test_batch_categorization(self):
        txns = [
            {"description": "netflix streaming service", "amount": 15.0},
            {"description": "walmart groceries", "amount": 80.0},
        ]
        results = run(self.cat.categorize_batch(txns))
        assert len(results) == 2

    def test_statistics(self):
        stats = self.cat.get_category_statistics()
        assert "ml_available" in stats
        assert stats["total_patterns"] > 0


# ---------------------------------------------------------------------------
# Cognitive anomaly detection tests
# ---------------------------------------------------------------------------

class TestCognitiveAnomalyDetection:
    def setup_method(self):
        from financial.cognitive_analysis import CognitiveFinancialAnalyzer
        self.analyzer = CognitiveFinancialAnalyzer()

    def _make_transactions(self, amounts, category="groceries"):
        from datetime import datetime, timedelta
        base = datetime(2024, 1, 1)
        return [
            {
                "amount": a,
                "date": (base + timedelta(days=i)).isoformat(),
                "category": category,
                "description": f"txn {i}",
                "account": "checking"
            }
            for i, a in enumerate(amounts)
        ]

    def test_statistical_anomaly_detects_outlier(self):
        amounts = [50.0] * 20 + [1000.0]  # one clear outlier
        data = self._make_transactions(amounts)
        processed = self.analyzer._preprocess_transactions(data)
        anomalies = run(self.analyzer._statistical_anomaly_detection(processed, 0.8))
        assert len(anomalies) >= 1
        flagged_amounts = [a["transaction"]["amount"] for a in anomalies]
        assert 1000.0 in flagged_amounts

    def test_no_false_positives_on_uniform_data(self):
        amounts = [50.0] * 30
        data = self._make_transactions(amounts)
        processed = self.analyzer._preprocess_transactions(data)
        anomalies = run(self.analyzer._statistical_anomaly_detection(processed, 0.8))
        assert len(anomalies) == 0  # uniform data has no outliers

    def test_behavioral_anomaly_detection(self):
        normal = [30.0] * 15
        spike = [500.0]
        amounts = normal + spike
        data = self._make_transactions(amounts)
        processed = self.analyzer._preprocess_transactions(data)
        anomalies = run(self.analyzer._behavioral_anomaly_detection(processed))
        assert len(anomalies) >= 1

    def test_full_detect_anomalies(self):
        amounts = [50.0] * 20 + [2000.0]
        data = self._make_transactions(amounts)
        result = run(self.analyzer.detect_anomalies(data, sensitivity=0.8))
        assert "anomaly_summary" in result
        assert result["anomaly_summary"]["total_anomalies"] >= 1


# ---------------------------------------------------------------------------
# NL query interface tests
# ---------------------------------------------------------------------------

class TestNLQueryInterface:
    def setup_method(self):
        from financial.nl_query_interface import NaturalLanguageQueryEngine
        self.engine = NaturalLanguageQueryEngine()

    def test_parse_spending_intent(self):
        intent = run(self.engine.parse_query("how much did I spend on groceries this month?"))
        from financial.nl_query_interface import QueryType
        assert intent.query_type == QueryType.SPENDING

    def test_parse_balance_intent(self):
        intent = run(self.engine.parse_query("what is my current balance?"))
        from financial.nl_query_interface import QueryType
        assert intent.query_type == QueryType.BALANCE

    def test_parse_summary_intent(self):
        intent = run(self.engine.parse_query("give me a summary for this month"))
        from financial.nl_query_interface import QueryType
        assert intent.query_type == QueryType.SUMMARY

    def test_execute_balance_no_gnucash(self):
        """Without GnuCash connection, should return mock data."""
        result = run(self.engine.execute_query("what is my balance?"))
        assert result["success"]
        assert "total_balance" in result["result"]

    def test_execute_spending_no_gnucash(self):
        result = run(self.engine.execute_query("how much did I spend this month?"))
        assert result["success"]
        assert "total_spending" in result["result"]

    def test_timeframe_extraction_this_month(self):
        intent = run(self.engine.parse_query("spending this month"))
        assert intent.timeframe is not None

    def test_query_suggestions(self):
        suggestions = run(self.engine.get_query_suggestions("how"))
        assert isinstance(suggestions, list)
        assert all("how" in s.lower() for s in suggestions)
