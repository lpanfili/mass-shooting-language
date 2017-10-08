import json

with open('extraction_rules.json', 'r') as rules_file:
  RULES = json.load(rules_file)

def getRulesForDomain(domain):
  if domain in RULES:
    return RULES[domain]

def domainHasRules(domain):
  return domain in RULES

def getArticleSelectorsForDomain(domain):
  if domain in RULES:
    return RULES[domain]["article_text_selectors"]