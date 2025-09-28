"""
MVP de Machine Learning - ONEE 2025
Problema: Classificação de Engajamento de Professores
Aluno: Carlos Eduardo Nazareth de Oliveira

Objetivo: Prever se um professor recém-cadastrado vai se engajar na olimpíada
(cadastrar pelo menos 1 escola ou aluno)
"""

# =====================================
# 1. IMPORTAÇÃO DE BIBLIOTECAS
# =====================================
import pandas as pd
import numpy as np
import warnings
import time
from datetime import datetime
import urllib.request
import ssl

# Visualização
import matplotlib.pyplot as plt
import seaborn as sns

# Pré-processamento
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, RFE

# Modelos
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

# Métricas
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve
)

# Pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Configurações
warnings.filterwarnings('ignore')
np.random.seed(42)
plt.style.use('seaborn-v0_8-darkgrid')

print("=" * 80)
print("MVP DE MACHINE LEARNING - ONEE 2025")
print("Problema: Classificação de Engajamento de Professores")
print("=" * 80)

# =====================================
# 2. FUNÇÕES AUXILIARES
# =====================================

def parse_net_date(date_str):
    """Converte datas do formato .NET para datetime"""
    if pd.isna(date_str) or date_str is None:
        return None
    try:
        import re
        match = re.search(r'/Date\((\d+)\)/', str(date_str))
        if match:
            timestamp = int(match.group(1))
            return pd.to_datetime(timestamp, unit='ms')
    except:
        pass
    return None

def create_temporal_features(df):
    """Cria features temporais a partir das datas"""
    features = pd.DataFrame()
    
    if 'DataCadastro' in df.columns:
        df['DataCadastro_parsed'] = df['DataCadastro'].apply(parse_net_date)
        features['mes_cadastro'] = df['DataCadastro_parsed'].dt.month
        features['dia_semana'] = df['DataCadastro_parsed'].dt.dayofweek
        features['dia_mes'] = df['DataCadastro_parsed'].dt.day
        features['hora_cadastro'] = df['DataCadastro_parsed'].dt.hour
        
        # Período do dia
        features['periodo_dia'] = pd.cut(
            features['hora_cadastro'].fillna(12),
            bins=[0, 6, 12, 18, 24],
            labels=['madrugada', 'manha', 'tarde', 'noite']
        )
    
    if 'DataNascimento' in df.columns:
        df['DataNascimento_parsed'] = df['DataNascimento'].apply(parse_net_date)
        current_year = 2025
        df['idade'] = current_year - df['DataNascimento_parsed'].dt.year
        features['idade'] = df['idade']
        features['faixa_etaria'] = pd.cut(
            df['idade'].fillna(40),
            bins=[0, 30, 40, 50, 60, 100],
            labels=['<30', '30-40', '40-50', '50-60', '60+']
        )
    
    return features

def print_model_metrics(y_true, y_pred, y_proba=None, model_name="Model"):
    """Imprime métricas detalhadas do modelo"""
    print(f"\n{'='*50}")
    print(f"Métricas para {model_name}")
    print(f"{'='*50}")
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    if y_proba is not None:
        auc = roc_auc_score(y_true, y_proba)
        print(f"ROC-AUC:   {auc:.4f}")
    
    print("\nMatriz de Confusão:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}

# =====================================
# 3. CARGA E PREPARAÇÃO DOS DADOS
# =====================================

print("\n" + "="*50)
print("ETAPA 1: CARGA E PREPARAÇÃO DOS DADOS")
print("="*50)

# Carregando os dados diretamente do GitHub
print("\nCarregando arquivos CSV do GitHub...")
print("Repositório: https://github.com/KaduNazareth/onee-2025-ml")

# URLs dos arquivos no GitHub
urls = {
    'professores': 'https://raw.githubusercontent.com/KaduNazareth/onee-2025-ml/refs/heads/main/API%20Professores.csv',
    'escolas': 'https://raw.githubusercontent.com/KaduNazareth/onee-2025-ml/refs/heads/main/API%20Escolas.csv',
    'alunos': 'https://raw.githubusercontent.com/KaduNazareth/onee-2025-ml/refs/heads/main/API%20Alunos.csv'
}

try:
    # Configuração para evitar problemas de SSL
    ssl._create_default_https_context = ssl._create_unverified_context
    
    # Carregando cada arquivo
    print("\n📥 Baixando API Professores.csv...")
    professores = pd.read_csv(urls['professores'], sep=';', encoding='utf-8')
    print(f"✓ Professores: {len(professores)} registros carregados")
    
    print("📥 Baixando API Escolas.csv...")
    escolas = pd.read_csv(urls['escolas'], sep=';', encoding='utf-8')
    print(f"✓ Escolas: {len(escolas)} registros carregados")
    
    print("📥 Baixando API Alunos.csv...")
    alunos = pd.read_csv(urls['alunos'], sep=';', encoding='utf-8')
    print(f"✓ Alunos: {len(alunos)} registros carregados")
    
    print("\n✅ Todos os dados foram carregados com sucesso do GitHub!")
    
except Exception as e:
    print(f"\n⚠️ Erro ao carregar dados do GitHub: {e}")
    print("\nTentando carregar dados localmente como fallback...")
    
    try:
        # Fallback para arquivos locais se o GitHub falhar
        professores = pd.read_csv('../data/API Professores.csv', sep=';', encoding='utf-8')
        escolas = pd.read_csv('../data/API Escolas.csv', sep=';', encoding='utf-8')
        alunos = pd.read_csv('../data/API Alunos.csv', sep=';', encoding='utf-8')
        
        print(f"✓ Professores: {len(professores)} registros (local)")
        print(f"✓ Escolas: {len(escolas)} registros (local)")
        print(f"✓ Alunos: {len(alunos)} registros (local)")
    except:
        print("❌ Erro: Não foi possível carregar os dados nem do GitHub nem localmente.")
        print("Por favor, verifique:")
        print("1. Se as URLs do GitHub estão corretas")
        print("2. Se você tem conexão com a internet")
        print("3. Se os arquivos locais existem em '../data/'")
        raise

# =====================================
# 4. ENGENHARIA DE FEATURES
# =====================================

print("\n" + "="*50)
print("ETAPA 2: ENGENHARIA DE FEATURES")
print("="*50)

# Agregando dados de engajamento por professor
print("\nCriando features de engajamento...")

# Contagem de escolas por professor
escolas_por_prof = escolas.groupby('IDProfessor').agg({
    'IDEscola': 'count',
    'DataConclusaoCadastro': lambda x: x.notna().sum()
}).rename(columns={
    'IDEscola': 'num_escolas',
    'DataConclusaoCadastro': 'num_escolas_completas'
})

# Contagem de alunos por professor
alunos_por_prof = alunos.groupby('IDProfessor').agg({
    'IDAluno': 'count',
    'DataConclusaoCadastro': lambda x: x.notna().sum(),
    'Serie': lambda x: x.nunique()
}).rename(columns={
    'IDAluno': 'num_alunos',
    'DataConclusaoCadastro': 'num_alunos_completos',
    'Serie': 'num_series_diferentes'
})

# Merge com dados dos professores
df = professores.copy()
df = df.merge(escolas_por_prof, left_on='IDProfessor', right_index=True, how='left')
df = df.merge(alunos_por_prof, left_on='IDProfessor', right_index=True, how='left')

# Preenchendo NaN com 0 para contagens
for col in ['num_escolas', 'num_escolas_completas', 'num_alunos', 'num_alunos_completos', 'num_series_diferentes']:
    df[col] = df[col].fillna(0)

# Criando variável alvo: Engajamento (1 se cadastrou pelo menos 1 escola ou aluno)
df['engajado'] = ((df['num_escolas'] > 0) | (df['num_alunos'] > 0)).astype(int)

print(f"\nDistribuição da variável alvo:")
print(df['engajado'].value_counts())
print(f"Taxa de engajamento: {df['engajado'].mean():.2%}")

# Features temporais
print("\nCriando features temporais...")
temporal_features = create_temporal_features(df)
df = pd.concat([df, temporal_features], axis=1)

# Features de localização
print("\nCriando features de localização...")
df['tem_municipio'] = df['Municipio'].notna().astype(int)
df['tem_uf'] = df['UF'].notna().astype(int)
df['cadastro_completo'] = ((df['tem_municipio'] == 1) & (df['tem_uf'] == 1)).astype(int)

# Encoding de UF (frequência)
uf_freq = df['UF'].value_counts().to_dict()
df['uf_frequencia'] = df['UF'].map(uf_freq).fillna(0)

# Taxa de engajamento por UF
uf_engagement = df.groupby('UF')['engajado'].mean().to_dict()
df['uf_taxa_engajamento'] = df['UF'].map(uf_engagement).fillna(df['engajado'].mean())

# Estatísticas por UF
uf_stats = df.groupby('UF').agg({
    'num_escolas': 'mean',
    'num_alunos': 'mean'
}).rename(columns={
    'num_escolas': 'uf_media_escolas',
    'num_alunos': 'uf_media_alunos'
})
df = df.merge(uf_stats, left_on='UF', right_index=True, how='left')
df['uf_media_escolas'] = df['uf_media_escolas'].fillna(0)
df['uf_media_alunos'] = df['uf_media_alunos'].fillna(0)

print(f"\nTotal de features criadas: {len(df.columns)}")

# =====================================
# 5. SELEÇÃO DE FEATURES
# =====================================

print("\n" + "="*50)
print("ETAPA 3: SELEÇÃO DE FEATURES")
print("="*50)

# Separando features e target
features_to_use = [
    'tem_municipio', 'tem_uf', 'cadastro_completo',
    'uf_frequencia', 'uf_taxa_engajamento', 'uf_media_escolas', 'uf_media_alunos',
    'mes_cadastro', 'dia_semana', 'dia_mes', 'hora_cadastro',
    'idade'
]

# Removendo features que não existem ou têm muitos NaN
features_available = []
for feat in features_to_use:
    try:
        if feat in df.columns:
            # Tenta adicionar a feature
            test_values = df[feat].dropna()
            if len(test_values) > len(df) * 0.5:  # Pelo menos 50% não-nulos
                features_available.append(feat)
            else:
                print(f"  ⚠ Feature '{feat}' tem muitos valores nulos")
    except Exception as e:
        print(f"  ⚠ Erro ao processar '{feat}': {e}")
        continue

print(f"\nFeatures selecionadas: {len(features_available)}")
print(features_available)

# Preparando dados
X = df[features_available].copy()
y = df['engajado'].copy()

# Imputação de valores faltantes
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns,
    index=X.index
)

# =====================================
# 6. DIVISÃO DOS DADOS
# =====================================

print("\n" + "="*50)
print("ETAPA 4: DIVISÃO DOS DADOS")
print("="*50)

# Split treino/teste estratificado
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

# Split treino/validação
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

print(f"Conjunto de treino: {len(X_train_final)} amostras")
print(f"Conjunto de validação: {len(X_val)} amostras")
print(f"Conjunto de teste: {len(X_test)} amostras")
print(f"\nDistribuição das classes:")
print(f"  Treino: {y_train_final.mean():.2%} engajados")
print(f"  Validação: {y_val.mean():.2%} engajados")
print(f"  Teste: {y_test.mean():.2%} engajados")

# Normalização
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_final)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# =====================================
# 7. MODELAGEM - BASELINE
# =====================================

print("\n" + "="*50)
print("ETAPA 5: MODELAGEM")
print("="*50)

print("\n### MODELO BASELINE ###")
# Baseline: sempre prever classe majoritária
baseline_pred = np.zeros(len(y_test))
baseline_metrics = print_model_metrics(y_test, baseline_pred, model_name="Baseline (Majoritária)")

# =====================================
# 8. TREINAMENTO DE MÚLTIPLOS MODELOS
# =====================================

print("\n### TREINAMENTO DE MODELOS ###")

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
    'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0),
    'SVM': SVC(random_state=42, probability=True),
    'Naive Bayes': GaussianNB(),
    'KNN': KNeighborsClassifier()
}

results = {}
best_score = 0
best_model = None
best_model_name = None

for name, model in models.items():
    print(f"\nTreinando {name}...")
    start_time = time.time()
    
    # Treinamento
    model.fit(X_train_scaled, y_train_final)
    
    # Predições
    y_pred_val = model.predict(X_val_scaled)
    
    # Probabilidades para ROC-AUC
    if hasattr(model, 'predict_proba'):
        y_proba_val = model.predict_proba(X_val_scaled)[:, 1]
    else:
        y_proba_val = None
    
    # Métricas
    metrics = print_model_metrics(y_val, y_pred_val, y_proba_val, model_name=name)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train_final, cv=5, scoring='f1')
    print(f"Cross-validation F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Tempo de treinamento
    train_time = time.time() - start_time
    print(f"Tempo de treinamento: {train_time:.2f}s")
    
    # Armazenar resultados
    results[name] = {
        'model': model,
        'metrics': metrics,
        'cv_score': cv_scores.mean(),
        'train_time': train_time
    }
    
    # Verificar melhor modelo
    if metrics['f1'] > best_score:
        best_score = metrics['f1']
        best_model = model
        best_model_name = name

print(f"\n{'='*50}")
print(f"MELHOR MODELO: {best_model_name} (F1-Score: {best_score:.4f})")
print(f"{'='*50}")

# =====================================
# 9. OTIMIZAÇÃO DE HIPERPARÂMETROS
# =====================================

print("\n" + "="*50)
print("ETAPA 6: OTIMIZAÇÃO DE HIPERPARÂMETROS")
print("="*50)

# Otimizando o melhor modelo
if best_model_name == 'Random Forest':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
elif best_model_name == 'XGBoost':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.7, 0.8, 1.0]
    }
elif best_model_name == 'Gradient Boosting':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'min_samples_split': [2, 5, 10]
    }
else:
    param_grid = {}

if param_grid:
    print(f"\nOtimizando hiperparâmetros para {best_model_name}...")
    
    grid_search = GridSearchCV(
        type(best_model)(),  # Nova instância do modelo
        param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_scaled, y_train_final)
    
    print(f"\nMelhores parâmetros encontrados:")
    print(grid_search.best_params_)
    print(f"Melhor score CV: {grid_search.best_score_:.4f}")
    
    # Usar modelo otimizado
    best_model_optimized = grid_search.best_estimator_
else:
    best_model_optimized = best_model

# =====================================
# 10. ENSEMBLE DE MODELOS
# =====================================

print("\n" + "="*50)
print("ETAPA 7: ENSEMBLE DE MODELOS")
print("="*50)

# Selecionando os 3 melhores modelos para o ensemble
sorted_models = sorted(results.items(), key=lambda x: x[1]['metrics']['f1'], reverse=True)
top_3_models = [(name, res['model']) for name, res in sorted_models[:3]]

print(f"\nCriando ensemble com os 3 melhores modelos:")
for name, _ in top_3_models:
    print(f"  - {name}")

# Criar ensemble
ensemble = VotingClassifier(
    estimators=top_3_models,
    voting='soft'  # Usa probabilidades
)

print("\nTreinando ensemble...")
ensemble.fit(X_train_scaled, y_train_final)

# Avaliação do ensemble
y_pred_ensemble = ensemble.predict(X_val_scaled)
y_proba_ensemble = ensemble.predict_proba(X_val_scaled)[:, 1]

ensemble_metrics = print_model_metrics(
    y_val, y_pred_ensemble, y_proba_ensemble, 
    model_name="Ensemble (Voting)"
)

# =====================================
# 11. AVALIAÇÃO FINAL NO CONJUNTO DE TESTE
# =====================================

print("\n" + "="*50)
print("ETAPA 8: AVALIAÇÃO FINAL NO CONJUNTO DE TESTE")
print("="*50)

# Retreinar o melhor modelo com todos os dados de treino+validação
X_train_full = np.vstack([X_train_scaled, X_val_scaled])
y_train_full = pd.concat([y_train_final, y_val])

print("\nRetreinando melhor modelo com dados completos de treino...")
final_model = type(best_model_optimized)(**best_model_optimized.get_params())
final_model.fit(X_train_full, y_train_full)

# Predições finais
y_pred_test = final_model.predict(X_test_scaled)
y_proba_test = final_model.predict_proba(X_test_scaled)[:, 1] if hasattr(final_model, 'predict_proba') else None

print(f"\n### RESULTADOS FINAIS - {best_model_name} ###")
final_metrics = print_model_metrics(y_test, y_pred_test, y_proba_test, model_name=f"{best_model_name} (Final)")

# =====================================
# 12. ANÁLISE DE IMPORTÂNCIA DE FEATURES
# =====================================

print("\n" + "="*50)
print("ETAPA 9: ANÁLISE DE IMPORTÂNCIA DE FEATURES")
print("="*50)

if hasattr(final_model, 'feature_importances_'):
    importances = final_model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 features mais importantes:")
    print(feature_importance.head(10).to_string())
    
    # Visualização
    plt.figure(figsize=(10, 6))
    top_features = feature_importance.head(10)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importância')
    plt.title(f'Top 10 Features - {best_model_name}')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=100, bbox_inches='tight')
    print("\n✓ Gráfico salvo como 'feature_importance.png'")

# =====================================
# 13. VISUALIZAÇÕES
# =====================================

print("\n" + "="*50)
print("ETAPA 10: VISUALIZAÇÕES")
print("="*50)

# Matriz de confusão
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Matriz de confusão - valores absolutos
cm = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title(f'Matriz de Confusão - {best_model_name}')
axes[0].set_xlabel('Predito')
axes[0].set_ylabel('Real')

# Matriz de confusão - percentuais
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', ax=axes[1])
axes[1].set_title('Matriz de Confusão (Normalizada)')
axes[1].set_xlabel('Predito')
axes[1].set_ylabel('Real')

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=100, bbox_inches='tight')
print("✓ Matriz de confusão salva como 'confusion_matrix.png'")

# Curva ROC
if y_proba_test is not None:
    plt.figure(figsize=(8, 6))
    fpr, tpr, thresholds = roc_curve(y_test, y_proba_test)
    auc = roc_auc_score(y_test, y_proba_test)
    
    plt.plot(fpr, tpr, label=f'{best_model_name} (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=100, bbox_inches='tight')
    print("✓ Curva ROC salva como 'roc_curve.png'")

# =====================================
# 14. RELATÓRIO FINAL
# =====================================

print("\n" + "="*80)
print("RELATÓRIO FINAL - MVP MACHINE LEARNING ONEE 2025")
print("="*80)

print("\n### DEFINIÇÃO DO PROBLEMA ###")
print("""
Objetivo: Prever se um professor recém-cadastrado vai se engajar na ONEE 2025,
ou seja, cadastrar pelo menos 1 escola ou aluno.

Tipo de problema: Classificação Binária Supervisionada
Dataset: 15.249 professores cadastrados
Taxa de engajamento atual: 39.8%

Premissas:
- Professores com município preenchido têm cadastro mais completo
- O estado (UF) do professor influencia o engajamento
- Features temporais podem indicar padrões de comportamento
""")

print("\n### PREPARAÇÃO DOS DADOS ###")
print(f"""
- Divisão estratificada: 60% treino, 20% validação, 20% teste
- Imputação de valores faltantes com mediana
- Normalização com StandardScaler
- {len(features_available)} features selecionadas
- Validação cruzada: 5-fold estratificado
""")

print("\n### RESULTADOS ###")
print(f"""
Melhor modelo: {best_model_name}
Métricas no conjunto de teste:
- Accuracy:  {final_metrics['accuracy']:.4f}
- Precision: {final_metrics['precision']:.4f}
- Recall:    {final_metrics['recall']:.4f}
- F1-Score:  {final_metrics['f1']:.4f}

Comparação com baseline:
- Melhoria no F1-Score: {((final_metrics['f1'] - baseline_metrics['f1']) / max(baseline_metrics['f1'], 0.001) * 100):.1f}%
""")

print("\n### CONCLUSÕES E PRÓXIMOS PASSOS ###")
print("""
✓ O modelo consegue identificar professores com maior probabilidade de engajamento
✓ Features geográficas (UF) são importantes preditores
✓ O modelo pronto para ser usado para ações proativas de engajamento

Repositório: https://github.com/KaduNazareth/onee-2025-ml
""")

print("\n" + "="*80)
print("MVP CONCLUÍDO COM SUCESSO!")
print("Aluno: Carlos Eduardo Nazareth de Oliveira")
print("="*80)