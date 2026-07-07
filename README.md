# Mon premier RAG — Mini-TP

Master 2 MD5 — Data & IA

Un RAG minimal mais complet : ChromaDB + sentence-transformers + Groq + agent modérateur.

## Architecture

- `vector_db.py` — Classe `VectorDB` : création/rechargement d'une base ChromaDB persistée,
  encodage des chunks avec sentence-transformers, recherche des k plus proches voisins.
- `moderator.py` — Classe `Moderator` : détection de tentatives de prompt injection via un
  modèle de classification dédié, sortie JSON stricte.
- `rag.py` — Classe `RAG` : orchestre le pipeline complet (modération → retrieval → prompt →
  appel LLM).
- `config.py` — Noms des modèles (embedding et LLM), centralisés à un seul endroit.
- `prompts/` — Prompts système en fichiers texte, séparés du code.

## Installation

\`\`\`bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
\`\`\`

Créer un fichier `.env` à la racine avec :
\`\`\`
GROQ_API_KEY=Crée votre propre clé d'API sur https://platform.groq.ai et copiez-la ici.
\`\`\`

## Modèles utilisés

- Embedding : `distiluse-base-multilingual-cased-v2` (multilingue, léger)
- Génération : `llama-3.3-70b-versatile` (via Groq)
- Modération : famille `safeguard` (via Groq)

## Détail technique clé : la métadonnée de collection

Le nom du modèle d'embedding est stocké dans les métadonnées de la collection ChromaDB
elle-même, et non relu depuis `config.py` au rechargement. Cela évite un bug silencieux :
si `config.py` change de modèle après indexation, les nouveaux vecteurs de question ne
vivraient plus dans le même espace géométrique que les vecteurs déjà indexés — sans erreur
visible, juste une dégradation progressive et inexpliquée de la pertinence des réponses.

## Résultats des tests de validation (section 6 du TP)

| Scénario | Comportement attendu | Résultat |
|---|---|---|
| Question piégée (injection + vraie question) | Bloquée par le modérateur avant tout appel au LLM principal 
| Modérateur désactivé | Observation : le LLM principal résiste partiellement grâce au prompt strict, mais sans garantie observé |
| Question légitime hors corpus | Le système répond qu'il ne sait pas plutôt que d'inventer 
| Affirmation fausse de l'utilisateur | Le système signale la contradiction avec la version correcte du corpus 

## Ce que j'ai appris

- Séparer indexation et interrogation : ne jamais réindexer à chaque lancement.
- Toujours utiliser le même modèle d'embedding pour indexer et pour interroger.
- Tester le retrieval seul, avant de brancher le LLM (chunks mauvais = problème
  d'embedding/chunking, pas de prompt).
- Confier la modération à un modèle dédié plutôt qu'à une simple consigne dans le prompt
  principal, pour la fiabilité, la traçabilité et la séparation des responsabilités.