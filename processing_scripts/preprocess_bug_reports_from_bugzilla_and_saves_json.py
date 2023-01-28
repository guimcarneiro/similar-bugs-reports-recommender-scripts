import json

def is_change_to_bug_resolved(change):
  return (change["field_name"] == "status" and change["added"] == "RESOLVED")

def is_change_to_final_assigned_to(change, final_assigned_to):
  return (change["field_name"] == "assigned_to" and change["added"] == final_assigned_to)

# recupera data de modificação para RESOLVED
def retrieve_resolved_date(bug):
  # percorrer de trás para frente histórico, acessando sequência de mudanças
  # reverso pois está em ordem cronológica

  bug_history = bug["history"]

  for i in range(len(bug_history)-1, -1, -1):
    modification = bug_history[i]

    when_modified = modification["when"]

    changes = modification["changes"]

    for c in changes:
      if is_change_to_bug_resolved(c):
        return when_modified

  return ''

# recupera data de modificação para o assigned_to do responsável final
# se não tiver data de atribuição ao assigned_to final, atribui data de criação
def retrieve_when_assigned_to_final_dev(bug):
  bug_history = bug["history"]

  creation_date = bug["creation_time"]

  final_assigned_to = bug["assigned_to"]

  for i in range(len(bug_history)-1, -1, -1):
    modification = bug_history[i]

    when_modified = modification["when"]

    changes = modification["changes"]

    for c in changes:
      if is_change_to_final_assigned_to(c, final_assigned_to):
        return when_modified
  return creation_date

# Bug reports crus baixados do Bugzilla
with open('data/bug_reports_base_2009_2012_nextbug.json') as input:
  raw_bugs = json.load(input)

# atribui 2 colunas novas
for bug in raw_bugs:
  bug["when_changed_to_resolved"] = retrieve_resolved_date(bug)
  bug["when_final_change_assigned_to"] = retrieve_when_assigned_to_final_dev(bug)

# remove campo de histórico
for bug in raw_bugs:
  bug.pop("history")

# cria campo _id
for bug in raw_bugs:
  bug["bg_number"] = bug["id"]
  bug.pop("id")

# salva json tratado
with open("data/bug_reports_new_columns_no_history_field_2009_2012_nextbug.json", "w") as outfile:
  json.dump(raw_bugs, outfile)
