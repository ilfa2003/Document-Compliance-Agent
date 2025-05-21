USER_RULES = """
Match invoice LACO-39 to customs declaration 203-04144376-23 and consignment note 319303 using the respective fields.
Ensure invoice date is on or after customs certificate date.
Check total weight matches across all documents.
Check exporter name consistency.
Check vehicle/container numbers match.
Generate a compliance report with pass/fail for each rule and explanations.
Additionally, use any similar or related documents retrieved by vector search to inform your reasoning and catch edge cases or fuzzy matches.
""" 