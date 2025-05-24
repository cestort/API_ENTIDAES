#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script para descargar un modelo de Hugging Face y su tokenizador
en la carpeta 'models/' (o la que especifiques), con todos los archivos necesarios.
"""

import os
import argparse
from transformers import AutoTokenizer, AutoModelForTokenClassification

def download_model(model_id: str, output_dir: str):
    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Descargar y guardar el tokenizador
    print(f"🔄 Descargando tokenizador '{model_id}'…")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Tokenizador guardado en '{output_dir}'.")

    # Descargar y guardar el modelo
    print(f"🔄 Descargando modelo '{model_id}'…")
    model = AutoModelForTokenClassification.from_pretrained(model_id)
    model.save_pretrained(output_dir)
    print(f"✅ Modelo guardado en '{output_dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Descarga un modelo HF y su tokenizador en local"
    )
    parser.add_argument(
        "--model",
        default="xlm-roberta-large-finetuned-conll03-english",
        help="ID del modelo en Hugging Face (p.ej. dbmdz/bert-large-cased-finetuned-conll03-english)"
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Directorio donde se guardarán los archivos del modelo"
    )
    args = parser.parse_args()

    download_model(args.model, args.output_dir)
