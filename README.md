# MNIST – MLP y Multi-Head Attention con PyTorch Lightning

Este repositorio contiene una implementación completa para entrenar modelos en el dataset **MNIST**, usando dos enfoques:

1. **MLP tradicional**
2. **Modelo basado en Multi-Head Attention**, donde las imágenes se dividen en parches (patch embeddings) y se procesan con un mecanismo inspirado en arquitecturas tipo Transformer.

El proyecto está construido con **PyTorch Lightning**, lo que simplifica el entrenamiento y valida el uso de buenas prácticas como `LightningModule` y `LightningDataModule`.

---

🧩 Requisitos

Antes de ejecutar el script, instala las dependencias:

pip install -r requirements.txt

🧑‍💻 Autor

Desarrollado por Gus como parte de su aprendizaje en Python e IA.
