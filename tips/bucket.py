from google.cloud import aiplatform

aiplatform.init(project="votre-projet", location="us-central1")
models = aiplatform.GenaiModel.list()
print(models[0]._raw_model.name)  # Doit afficher "models/gemini-pro"
