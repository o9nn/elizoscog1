;;; CogServer Initialization Script
;;; 
;;; This script is loaded when the CogServer starts and configures
;;; the AtomSpace with financial reasoning capabilities.

(use-modules (opencog) (opencog exec) (opencog ure))
(use-modules (opencog cogserver))

;; Configure logging
(cog-logger-set-level! "info")
(cog-logger-set-filename! "/var/log/cogserver.log")

;; Load financial reasoning rules
(load "/opt/opencog/rules/financial-reasoning.scm")

;; Initialize financial knowledge categories
(ConceptNode "essential-expense")
(ConceptNode "discretionary-expense")
(ConceptNode "investment")
(ConceptNode "savings")
(ConceptNode "income")

;; Common expense category hierarchies
(InheritanceLink (stv 1.0 1.0)
  (ConceptNode "housing")
  (ConceptNode "essential-expense"))

(InheritanceLink (stv 1.0 1.0)
  (ConceptNode "utilities")
  (ConceptNode "essential-expense"))

(InheritanceLink (stv 1.0 1.0)
  (ConceptNode "groceries")
  (ConceptNode "essential-expense"))

(InheritanceLink (stv 1.0 1.0)
  (ConceptNode "healthcare")
  (ConceptNode "essential-expense"))

(InheritanceLink (stv 1.0 1.0)
  (ConceptNode "insurance")
  (ConceptNode "essential-expense"))

(InheritanceLink (stv 1.0 1.0)
  (ConceptNode "transportation")
  (ConceptNode "essential-expense"))

(InheritanceLink (stv 0.9 0.9)
  (ConceptNode "dining")
  (ConceptNode "discretionary-expense"))

(InheritanceLink (stv 1.0 1.0)
  (ConceptNode "entertainment")
  (ConceptNode "discretionary-expense"))

(InheritanceLink (stv 1.0 1.0)
  (ConceptNode "shopping")
  (ConceptNode "discretionary-expense"))

(InheritanceLink (stv 1.0 1.0)
  (ConceptNode "travel")
  (ConceptNode "discretionary-expense"))

(InheritanceLink (stv 1.0 1.0)
  (ConceptNode "subscriptions")
  (ConceptNode "discretionary-expense"))

;; Common merchant-to-category mappings
(InheritanceLink (stv 0.95 0.95)
  (ConceptNode "walmart")
  (ConceptNode "groceries"))

(InheritanceLink (stv 0.95 0.95)
  (ConceptNode "target")
  (ConceptNode "shopping"))

(InheritanceLink (stv 0.95 0.95)
  (ConceptNode "amazon")
  (ConceptNode "shopping"))

(InheritanceLink (stv 0.95 0.95)
  (ConceptNode "costco")
  (ConceptNode "groceries"))

(InheritanceLink (stv 0.95 0.95)
  (ConceptNode "netflix")
  (ConceptNode "subscriptions"))

(InheritanceLink (stv 0.95 0.95)
  (ConceptNode "spotify")
  (ConceptNode "subscriptions"))

(InheritanceLink (stv 0.95 0.95)
  (ConceptNode "uber")
  (ConceptNode "transportation"))

(InheritanceLink (stv 0.95 0.95)
  (ConceptNode "lyft")
  (ConceptNode "transportation"))

;; Risk level definitions
(ConceptNode "low-risk")
(ConceptNode "medium-risk")
(ConceptNode "high-risk")

;; Define risk level thresholds
(EvaluationLink
  (PredicateNode "risk-threshold")
  (ListLink
    (ConceptNode "low-risk")
    (NumberNode "0.3")))

(EvaluationLink
  (PredicateNode "risk-threshold")
  (ListLink
    (ConceptNode "medium-risk")
    (NumberNode "0.7")))

(display "\n=== CogServer Financial Reasoning Initialized ===\n")
(display "Available rule bases: financial-reasoning-rbs\n")
(display "Loaded expense categories and merchant mappings\n")
(display "Ready for cognitive financial analysis.\n\n")
