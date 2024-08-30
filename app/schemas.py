from pydantic import BaseModel

# Schéma pour l'entrée de texte brut
class TextInput(BaseModel):
    text: str
