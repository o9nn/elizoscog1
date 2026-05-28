;;; Financial Reasoning Rules for PLN (Probabilistic Logic Networks)
;;; 
;;; This file defines cognitive reasoning rules for financial analysis
;;; using OpenCog's Probabilistic Logic Networks framework.
;;;
;;; Rules are organized by category:
;;; 1. Transaction Pattern Rules
;;; 2. Risk Assessment Rules
;;; 3. Budget Planning Rules
;;; 4. Anomaly Detection Rules
;;; 5. Investment Reasoning Rules

(use-modules (opencog) (opencog exec) (opencog ure))

;;; ============================================================================
;;; CATEGORY 1: Transaction Pattern Rules
;;; ============================================================================

;; Rule: Identify recurring transaction patterns
;; If a transaction type appears with similar amounts at regular intervals,
;; infer it is a recurring expense/income
(define recurring-transaction-rule
  (BindLink
    (VariableList
      (TypedVariableLink
        (VariableNode "$transaction")
        (TypeNode "ConceptNode"))
      (TypedVariableLink
        (VariableNode "$category")
        (TypeNode "ConceptNode"))
      (TypedVariableLink
        (VariableNode "$amount")
        (TypeNode "NumberNode")))
    (AndLink
      (EvaluationLink
        (PredicateNode "transaction-category")
        (ListLink
          (VariableNode "$transaction")
          (VariableNode "$category")))
      (EvaluationLink
        (PredicateNode "transaction-amount")
        (ListLink
          (VariableNode "$transaction")
          (VariableNode "$amount")))
      (EvaluationLink
        (PredicateNode "appears-regularly")
        (VariableNode "$transaction")))
    (EvaluationLink
      (stv 0.9 0.8)  ; High confidence for recurring patterns
      (PredicateNode "is-recurring-expense")
      (ListLink
        (VariableNode "$category")
        (VariableNode "$amount")))))

;; Rule: Categorize transaction by merchant name patterns
(define merchant-categorization-rule
  (BindLink
    (VariableList
      (TypedVariableLink
        (VariableNode "$transaction")
        (TypeNode "ConceptNode"))
      (TypedVariableLink
        (VariableNode "$merchant")
        (TypeNode "ConceptNode"))
      (TypedVariableLink
        (VariableNode "$category")
        (TypeNode "ConceptNode")))
    (AndLink
      (EvaluationLink
        (PredicateNode "transaction-merchant")
        (ListLink
          (VariableNode "$transaction")
          (VariableNode "$merchant")))
      (InheritanceLink
        (VariableNode "$merchant")
        (VariableNode "$category")))
    (EvaluationLink
      (stv 0.85 0.75)
      (PredicateNode "transaction-category")
      (ListLink
        (VariableNode "$transaction")
        (VariableNode "$category")))))

;;; ============================================================================
;;; CATEGORY 2: Risk Assessment Rules
;;; ============================================================================

;; Rule: High expense-to-income ratio indicates liquidity risk
(define liquidity-risk-rule
  (BindLink
    (VariableList
      (TypedVariableLink
        (VariableNode "$period")
        (TypeNode "ConceptNode"))
      (TypedVariableLink
        (VariableNode "$expenses")
        (TypeNode "NumberNode"))
      (TypedVariableLink
        (VariableNode "$income")
        (TypeNode "NumberNode")))
    (AndLink
      (EvaluationLink
        (PredicateNode "period-expenses")
        (ListLink
          (VariableNode "$period")
          (VariableNode "$expenses")))
      (EvaluationLink
        (PredicateNode "period-income")
        (ListLink
          (VariableNode "$period")
          (VariableNode "$income")))
      (GreaterThanLink
        (TimesLink
          (VariableNode "$expenses")
          (NumberNode "1"))
        (TimesLink
          (VariableNode "$income")
          (NumberNode "0.9"))))  ; Expenses > 90% of income
    (EvaluationLink
      (stv 0.85 0.9)
      (PredicateNode "has-liquidity-risk")
      (VariableNode "$period"))))

;; Rule: Spending volatility indicates behavioral risk
(define volatility-risk-rule
  (BindLink
    (VariableList
      (TypedVariableLink
        (VariableNode "$category")
        (TypeNode "ConceptNode"))
      (TypedVariableLink
        (VariableNode "$variance")
        (TypeNode "NumberNode")))
    (AndLink
      (EvaluationLink
        (PredicateNode "spending-variance")
        (ListLink
          (VariableNode "$category")
          (VariableNode "$variance")))
      (GreaterThanLink
        (VariableNode "$variance")
        (NumberNode "0.5")))  ; High variance threshold
    (EvaluationLink
      (stv 0.8 0.85)
      (PredicateNode "has-volatility-risk")
      (VariableNode "$category"))))

;; Rule: Category concentration risk
(define concentration-risk-rule
  (BindLink
    (VariableList
      (TypedVariableLink
        (VariableNode "$category")
        (TypeNode "ConceptNode"))
      (TypedVariableLink
        (VariableNode "$percentage")
        (TypeNode "NumberNode")))
    (AndLink
      (EvaluationLink
        (PredicateNode "spending-percentage")
        (ListLink
          (VariableNode "$category")
          (VariableNode "$percentage")))
      (GreaterThanLink
        (VariableNode "$percentage")
        (NumberNode "50")))  ; More than 50% in one category
    (EvaluationLink
      (stv 0.75 0.8)
      (PredicateNode "has-concentration-risk")
      (VariableNode "$category"))))

;;; ============================================================================
;;; CATEGORY 3: Budget Planning Rules
;;; ============================================================================

;; Rule: Recommend budget based on historical spending
(define budget-recommendation-rule
  (BindLink
    (VariableList
      (TypedVariableLink
        (VariableNode "$category")
        (TypeNode "ConceptNode"))
      (TypedVariableLink
        (VariableNode "$avg-spending")
        (TypeNode "NumberNode"))
      (TypedVariableLink
        (VariableNode "$income")
        (TypeNode "NumberNode")))
    (AndLink
      (EvaluationLink
        (PredicateNode "average-monthly-spending")
        (ListLink
          (VariableNode "$category")
          (VariableNode "$avg-spending")))
      (EvaluationLink
        (PredicateNode "monthly-income")
        (VariableNode "$income"))
      (InheritanceLink
        (VariableNode "$category")
        (ConceptNode "essential-expense")))
    (EvaluationLink
      (stv 0.9 0.85)
      (PredicateNode "recommended-budget")
      (ListLink
        (VariableNode "$category")
        ;; Budget = 110% of average spending for essentials
        (TimesLink
          (VariableNode "$avg-spending")
          (NumberNode "1.1"))))))

;; Rule: Savings goal feasibility
(define savings-feasibility-rule
  (BindLink
    (VariableList
      (TypedVariableLink
        (VariableNode "$goal-amount")
        (TypeNode "NumberNode"))
      (TypedVariableLink
        (VariableNode "$timeline-months")
        (TypeNode "NumberNode"))
      (TypedVariableLink
        (VariableNode "$monthly-surplus")
        (TypeNode "NumberNode")))
    (AndLink
      (EvaluationLink
        (PredicateNode "savings-goal")
        (VariableNode "$goal-amount"))
      (EvaluationLink
        (PredicateNode "goal-timeline")
        (VariableNode "$timeline-months"))
      (EvaluationLink
        (PredicateNode "monthly-surplus")
        (VariableNode "$monthly-surplus"))
      (GreaterThanLink
        (TimesLink
          (VariableNode "$monthly-surplus")
          (VariableNode "$timeline-months"))
        (VariableNode "$goal-amount")))
    (EvaluationLink
      (stv 0.95 0.9)
      (PredicateNode "savings-goal-achievable")
      (VariableNode "$goal-amount"))))

;;; ============================================================================
;;; CATEGORY 4: Anomaly Detection Rules
;;; ============================================================================

;; Rule: Unusual transaction amount
(define unusual-amount-rule
  (BindLink
    (VariableList
      (TypedVariableLink
        (VariableNode "$transaction")
        (TypeNode "ConceptNode"))
      (TypedVariableLink
        (VariableNode "$amount")
        (TypeNode "NumberNode"))
      (TypedVariableLink
        (VariableNode "$category")
        (TypeNode "ConceptNode"))
      (TypedVariableLink
        (VariableNode "$avg")
        (TypeNode "NumberNode"))
      (TypedVariableLink
        (VariableNode "$std")
        (TypeNode "NumberNode")))
    (AndLink
      (EvaluationLink
        (PredicateNode "transaction-amount")
        (ListLink
          (VariableNode "$transaction")
          (VariableNode "$amount")))
      (EvaluationLink
        (PredicateNode "transaction-category")
        (ListLink
          (VariableNode "$transaction")
          (VariableNode "$category")))
      (EvaluationLink
        (PredicateNode "category-mean")
        (ListLink
          (VariableNode "$category")
          (VariableNode "$avg")))
      (EvaluationLink
        (PredicateNode "category-std")
        (ListLink
          (VariableNode "$category")
          (VariableNode "$std")))
      ;; Z-score > 2 (unusual)
      (GreaterThanLink
        (AbsLink
          (DivideLink
            (MinusLink
              (VariableNode "$amount")
              (VariableNode "$avg"))
            (VariableNode "$std")))
        (NumberNode "2")))
    (EvaluationLink
      (stv 0.9 0.95)
      (PredicateNode "is-anomaly")
      (ListLink
        (VariableNode "$transaction")
        (ConceptNode "unusual-amount")))))

;; Rule: Unusual timing anomaly
(define unusual-timing-rule
  (BindLink
    (VariableList
      (TypedVariableLink
        (VariableNode "$transaction")
        (TypeNode "ConceptNode"))
      (TypedVariableLink
        (VariableNode "$hour")
        (TypeNode "NumberNode")))
    (AndLink
      (EvaluationLink
        (PredicateNode "transaction-hour")
        (ListLink
          (VariableNode "$transaction")
          (VariableNode "$hour")))
      ;; Late night transactions (10 PM - 6 AM)
      (OrLink
        (GreaterThanLink
          (VariableNode "$hour")
          (NumberNode "22"))
        (LessThanLink
          (VariableNode "$hour")
          (NumberNode "6"))))
    (EvaluationLink
      (stv 0.7 0.8)
      (PredicateNode "is-anomaly")
      (ListLink
        (VariableNode "$transaction")
        (ConceptNode "unusual-timing")))))

;;; ============================================================================
;;; CATEGORY 5: Investment Reasoning Rules
;;; ============================================================================

;; Rule: Portfolio diversification assessment
(define diversification-rule
  (BindLink
    (VariableList
      (TypedVariableLink
        (VariableNode "$portfolio")
        (TypeNode "ConceptNode"))
      (TypedVariableLink
        (VariableNode "$asset-count")
        (TypeNode "NumberNode"))
      (TypedVariableLink
        (VariableNode "$sector-count")
        (TypeNode "NumberNode")))
    (AndLink
      (EvaluationLink
        (PredicateNode "asset-count")
        (ListLink
          (VariableNode "$portfolio")
          (VariableNode "$asset-count")))
      (EvaluationLink
        (PredicateNode "sector-count")
        (ListLink
          (VariableNode "$portfolio")
          (VariableNode "$sector-count")))
      (GreaterThanLink
        (VariableNode "$asset-count")
        (NumberNode "10"))
      (GreaterThanLink
        (VariableNode "$sector-count")
        (NumberNode "5")))
    (EvaluationLink
      (stv 0.85 0.9)
      (PredicateNode "is-well-diversified")
      (VariableNode "$portfolio"))))

;; Rule: Risk tolerance assessment
(define risk-tolerance-rule
  (BindLink
    (VariableList
      (TypedVariableLink
        (VariableNode "$user")
        (TypeNode "ConceptNode"))
      (TypedVariableLink
        (VariableNode "$age")
        (TypeNode "NumberNode"))
      (TypedVariableLink
        (VariableNode "$income-stability")
        (TypeNode "NumberNode"))
      (TypedVariableLink
        (VariableNode "$emergency-fund-months")
        (TypeNode "NumberNode")))
    (AndLink
      (EvaluationLink
        (PredicateNode "user-age")
        (ListLink
          (VariableNode "$user")
          (VariableNode "$age")))
      (EvaluationLink
        (PredicateNode "income-stability-score")
        (ListLink
          (VariableNode "$user")
          (VariableNode "$income-stability")))
      (EvaluationLink
        (PredicateNode "emergency-fund-months")
        (ListLink
          (VariableNode "$user")
          (VariableNode "$emergency-fund-months")))
      ;; Young, stable income, adequate emergency fund
      (LessThanLink
        (VariableNode "$age")
        (NumberNode "40"))
      (GreaterThanLink
        (VariableNode "$income-stability")
        (NumberNode "0.7"))
      (GreaterThanLink
        (VariableNode "$emergency-fund-months")
        (NumberNode "6")))
    (EvaluationLink
      (stv 0.8 0.85)
      (PredicateNode "risk-tolerance")
      (ListLink
        (VariableNode "$user")
        (ConceptNode "high-risk-tolerance")))))

;;; ============================================================================
;;; Rule Engine Configuration
;;; ============================================================================

;; Define the financial reasoning rule base
(define financial-reasoning-rbs
  (ConceptNode "financial-reasoning-rbs"))

;; Add rules to the rule base
(ure-add-rules financial-reasoning-rbs
  (list
    (cons recurring-transaction-rule (stv 1.0 1.0))
    (cons merchant-categorization-rule (stv 1.0 1.0))
    (cons liquidity-risk-rule (stv 1.0 1.0))
    (cons volatility-risk-rule (stv 1.0 1.0))
    (cons concentration-risk-rule (stv 1.0 1.0))
    (cons budget-recommendation-rule (stv 1.0 1.0))
    (cons savings-feasibility-rule (stv 1.0 1.0))
    (cons unusual-amount-rule (stv 1.0 1.0))
    (cons unusual-timing-rule (stv 1.0 1.0))
    (cons diversification-rule (stv 1.0 1.0))
    (cons risk-tolerance-rule (stv 1.0 1.0))))

;; Configure inference parameters
(ure-set-maximum-iterations financial-reasoning-rbs 100)
(ure-set-complexity-penalty financial-reasoning-rbs 0.1)

(display "Financial reasoning rules loaded successfully.\n")
