# MVP Machine Learning - ONEE 2025

## Sobre o Projeto
Modelo de Machine Learning para prever engajamento de professores na Olimpíada Nacional de Eficiência Energética 2025.

## Resultados
- **F1-Score**: 72.02%
- **ROC-AUC**: 82.18%
- **Recall**: 79.88%

## Estrutura dos Dados
- `API Professores.csv`: 15.249 professores cadastrados
- `API Escolas.csv`: 6.509 escolas inscritas
- `API Alunos.csv`: 227.194 alunos participantes

## Como Usar

### Local
```python
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
python mvp_onee_ml.py
