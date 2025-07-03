import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from datetime import datetime
import json

# Configuration de la page
st.set_page_config(
    page_title="Détecteur d'Effets d'Aubaine ANR-CORDIS",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        padding: 1rem 0;
        border-bottom: 2px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_full_data():
    """
    Charge les données complètes depuis le fichier chainage_score30_VF.xlsx
    """
    try:
        df = pd.read_excel('chainage_score30_VF_corrigé.xlsx')
        return df
    except FileNotFoundError:
        st.error("Fichier 'chainage_score30_VF.xlsx' non trouvé. Assurez-vous qu'il est dans le même répertoire que l'application.")
        return None
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier : {str(e)}")
        return None

def load_data_with_upload():
    """
    Permet soit de charger le fichier directement, soit d'uploader un fichier
    """
    df_direct = load_full_data()
    
    if df_direct is not None:
        st.success(f"Fichier 'chainage_score30_VF.xlsx' chargé automatiquement ({len(df_direct)} lignes)")
        return df_direct
    else:
        st.warning("Fichier non trouvé. Veuillez l'uploader ci-dessous.")
        uploaded_file = st.file_uploader(
            "Choisissez votre fichier Excel",
            type=['xlsx', 'xls'],
            help="Uploadez le fichier chainage_score30_VF.xlsx"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_excel(uploaded_file)
                st.success(f"Fichier uploadé avec succès! ({len(df)} lignes)")
                return df
            except Exception as e:
                st.error(f"Erreur lors de la lecture du fichier uploadé : {str(e)}")
                return None
        else:
            return None

def count_partners(siren_list):
    """
    Compte le nombre de partenaires dans une liste SIREN
    """
    if pd.isna(siren_list) or siren_list == '':
        return 0
    
    siren_str = str(siren_list).replace('[', '').replace(']', '').strip()
    if siren_str == '':
        return 0
    
    siren_count = len([s.strip() for s in siren_str.split(',') if s.strip()])
    return siren_count

def process_data_flat(df, seuil_score_eleve, seuil_partenaires, min_partenaires):
    """
    Traite le DataFrame pour détecter les effets d'aubaine
    """
    try:
        # Vérification des colonnes requises
        required_cols = ['bert_score', 'tfidf_score', 'edition_anr', 'call_year', 
                        'trl_anr', 'trl_cordis', 'pct_siren_match', 'list_siren_anr', 'list_siren_cordis']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Colonnes manquantes dans le fichier : {', '.join(missing_cols)}")
            return None
        
        # Nettoyage des données
        numeric_cols = ['bert_score', 'tfidf_score', 'edition_anr', 'call_year', 
                       'trl_anr', 'trl_cordis', 'pct_siren_match']
        
        df_clean = df.copy()
        for col in numeric_cols:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        df_clean = df_clean.dropna(subset=numeric_cols)
        df_filtered = df_clean.copy()
        
        if len(df_filtered) == 0:
            st.warning("Aucune donnée valide après nettoyage.")
            return None
        
        # Calcul du nombre de partenaires
        df_filtered['nb_partners_anr'] = df_filtered['list_siren_anr'].apply(count_partners)
        df_filtered['nb_partners_cordis'] = df_filtered['list_siren_cordis'].apply(count_partners)
        
        # DÉTECTION DES 4 TYPES D'AUBAINES
        
        # 1. Aubaine de similarité
        df_filtered['aubaine_similarite'] = (
            (df_filtered['bert_score'] > seuil_score_eleve) |
            (df_filtered['tfidf_score'] > seuil_score_eleve)
        ).astype(bool)
        
        # 2. Aubaine temporelle
        df_filtered['aubaine_temporelle'] = (
            (abs(df_filtered['edition_anr'] - df_filtered['call_year']) <= 1)
        ).astype(bool)
        
        # 3. Aubaine TRL
        df_filtered['aubaine_trl'] = (
            (df_filtered['trl_anr'] == df_filtered['trl_cordis'])
        ).astype(bool)
        
        # 4. Aubaine partenaire
        df_filtered['aubaine_partenaire'] = (
            (df_filtered['pct_siren_match'] > seuil_partenaires) &
            ((df_filtered['nb_partners_anr'] >= min_partenaires) | (df_filtered['nb_partners_cordis'] >= min_partenaires))
        ).astype(bool)
        
        # INDICATEURS GLOBAUX
        
        # Présence d'au moins une aubaine
        df_filtered['a_aubaine'] = (
            df_filtered['aubaine_similarite'] | 
            df_filtered['aubaine_temporelle'] | 
            df_filtered['aubaine_trl'] | 
            df_filtered['aubaine_partenaire']
        ).astype(bool)
        
        # Liste des aubaines détectées
        def get_aubaines_detectees(row):
            aubaines = []
            if row['aubaine_similarite']:
                aubaines.append('Similarité')
            if row['aubaine_temporelle']:
                aubaines.append('Temporelle')
            if row['aubaine_trl']:
                aubaines.append('TRL')
            if row['aubaine_partenaire']:
                aubaines.append('Partenaire')
            return ', '.join(aubaines) if aubaines else 'Aucune'
        
        df_filtered['aubaines_detectees'] = df_filtered.apply(get_aubaines_detectees, axis=1)
        
        # Nombre total d'aubaines par projet
        df_filtered['nb_aubaines'] = (
            df_filtered['aubaine_similarite'].astype(int) +
            df_filtered['aubaine_temporelle'].astype(int) +
            df_filtered['aubaine_trl'].astype(int) +
            df_filtered['aubaine_partenaire'].astype(int)
        )
        
        # Statistiques
        total_lignes = len(df_filtered)
        aubaine_similarite_count = df_filtered['aubaine_similarite'].sum()
        aubaine_temporelle_count = df_filtered['aubaine_temporelle'].sum()
        aubaine_trl_count = df_filtered['aubaine_trl'].sum()
        aubaine_partenaire_count = df_filtered['aubaine_partenaire'].sum()
        projets_avec_aubaine = len(df_filtered[df_filtered['a_aubaine']])
        aubaines_multiples = len(df_filtered[df_filtered['nb_aubaines'] > 1])
        
        st.info(f"""
        **Statistiques de Détection :**
        - **Total de lignes analysées** : {total_lignes}
        - **Aubaines Similarité** : {aubaine_similarite_count} ({aubaine_similarite_count/total_lignes*100:.1f}%)
        - **Aubaines Temporelles** : {aubaine_temporelle_count} ({aubaine_temporelle_count/total_lignes*100:.1f}%)
        - **Aubaines TRL** : {aubaine_trl_count} ({aubaine_trl_count/total_lignes*100:.1f}%)
        - **Aubaines Partenaires** : {aubaine_partenaire_count} ({aubaine_partenaire_count/total_lignes*100:.1f}%)
        - **Total projets avec aubaine** : {projets_avec_aubaine} ({projets_avec_aubaine/total_lignes*100:.1f}%)
        - **Projets avec aubaines multiples** : {aubaines_multiples}
        
        **Paramètres utilisés :**
        - Seuil score élevé : {seuil_score_eleve}
        - Seuil partenaires : {seuil_partenaires}%
        - Minimum partenaires : {min_partenaires}
        """)
        
        return df_filtered
        
    except Exception as e:
        st.error(f"Erreur lors du traitement des données : {str(e)}")
        return None

def create_summary_stats_flat(df):
    """
    Crée les statistiques récapitulatives
    """
    if df is None or len(df) == 0:
        return None
    
    stats = {
        'total_projets': len(df),
        'aubaine_similarite': df['aubaine_similarite'].sum(),
        'aubaine_temporelle': df['aubaine_temporelle'].sum(),
        'aubaine_trl': df['aubaine_trl'].sum(),
        'aubaine_partenaire': df['aubaine_partenaire'].sum(),
        'projets_avec_aubaine': len(df[df['a_aubaine']]),
        'aubaines_multiples': len(df[df['nb_aubaines'] > 1]),
        'aubaines_doubles': len(df[df['nb_aubaines'] == 2]),
        'aubaines_triples': len(df[df['nb_aubaines'] == 3]),
        'aubaines_quadruples': len(df[df['nb_aubaines'] == 4])
    }
    
    return stats

def create_visualizations_flat(df):
    """
    Crée les visualisations
    """
    if df is None or len(df) == 0:
        return None, None, None, None
    
    # Graphique 1: Distribution par type d'aubaine
    types_counts = {
        'Similarité': df['aubaine_similarite'].sum(),
        'Temporelle': df['aubaine_temporelle'].sum(),
        'TRL': df['aubaine_trl'].sum(),
        'Partenaire': df['aubaine_partenaire'].sum()
    }
    
    fig1 = px.bar(
        x=list(types_counts.keys()),
        y=list(types_counts.values()),
        title="Distribution par Type d'Aubaine",
        labels={'x': 'Type d\'aubaine', 'y': 'Nombre de projets'},
        color=list(types_counts.values()),
        color_continuous_scale='Blues'
    )
    fig1.update_layout(showlegend=False, height=400)
    
    # Graphique 2: Distribution du nombre d'aubaines par projet
    nb_aubaines_counts = df['nb_aubaines'].value_counts().sort_index()
    
    fig2 = px.bar(
        x=nb_aubaines_counts.index,
        y=nb_aubaines_counts.values,
        title="Nombre d'Aubaines par Projet",
        labels={'x': 'Nombre d\'aubaines', 'y': 'Nombre de projets'},
        color=nb_aubaines_counts.values,
        color_continuous_scale='Viridis'
    )
    fig2.update_layout(showlegend=False)
    fig2.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=1))
    
    # Graphique 3: Scatter plot des scores
    df_plot = df.copy()
    df_plot['statut'] = df_plot['a_aubaine'].map({True: 'Avec aubaine', False: 'Sans aubaine'})
    
    fig3 = px.scatter(
        df_plot,
        x='bert_score',
        y='tfidf_score',
        color='statut',
        size='nb_aubaines',
        title="Scores BERT vs TF-IDF",
        labels={'bert_score': 'Score BERT', 'tfidf_score': 'Score TF-IDF'},
        hover_data=['code_projet_anr', 'acronyme_anr', 'aubaines_detectees'],
        color_discrete_map={
            'Avec aubaine': '#1f77b4',
            'Sans aubaine': '#d3d3d3'
        }
    )
    
    # Graphique 4: Évolution temporelle
    if 'edition_anr' in df.columns:
        temporal_data = df.groupby(['edition_anr', 'a_aubaine']).size().reset_index(name='count')
        temporal_data['statut'] = temporal_data['a_aubaine'].map({True: 'Avec aubaine', False: 'Sans aubaine'})
        
        fig4 = px.bar(
            temporal_data,
            x='edition_anr',
            y='count',
            color='statut',
            title="Évolution Temporelle des Aubaines",
            labels={'edition_anr': 'Édition ANR', 'count': 'Nombre de projets'},
            color_discrete_map={
                'Avec aubaine': '#1f77b4',
                'Sans aubaine': '#d3d3d3'
            }
        )
    else:
        # Heatmap des corrélations entre types
        aubaine_matrix = df[['aubaine_similarite', 'aubaine_temporelle', 'aubaine_trl', 'aubaine_partenaire']].astype(int)
        correlation_matrix = aubaine_matrix.corr()
        
        fig4 = px.imshow(
            correlation_matrix,
            title="Corrélation entre Types d'Aubaines",
            labels=dict(color="Corrélation"),
            color_continuous_scale="RdBu_r"
        )
        
        for i in range(len(correlation_matrix.index)):
            for j in range(len(correlation_matrix.columns)):
                fig4.add_annotation(
                    x=j, y=i,
                    text=f"{correlation_matrix.iloc[i, j]:.2f}",
                    showarrow=False,
                    font=dict(color="black" if abs(correlation_matrix.iloc[i, j]) < 0.5 else "white")
                )
    
    return fig1, fig2, fig3, fig4

def afficher_detection_aubaine_flat(df_original, seuil_score_eleve, seuil_partenaires, min_partenaires):
    """
    Affiche la détection d'aubaine
    """
    with st.spinner("Traitement des données en cours..."):
        df_processed = process_data_flat(df_original, seuil_score_eleve, seuil_partenaires, min_partenaires)
    
    if df_processed is not None:
        stats = create_summary_stats_flat(df_processed)
        
        st.header("STATISTIQUES - Récapitulatif")
        
        # Métriques principales
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Projets", stats['total_projets'])
        with col2:
            total_aubaines = stats['projets_avec_aubaine']
            pct_aubaines = (total_aubaines / stats['total_projets'] * 100) if stats['total_projets'] > 0 else 0
            st.metric("Projets avec Aubaine", f"{total_aubaines}", f"{pct_aubaines:.1f}%")
        with col3:
            pct_multiples = (stats['aubaines_multiples'] / stats['total_projets'] * 100) if stats['total_projets'] > 0 else 0
            st.metric("Aubaines Multiples", stats['aubaines_multiples'], f"{pct_multiples:.1f}%")
        with col4:
            projets_sans_aubaine = stats['total_projets'] - stats['projets_avec_aubaine']
            pct_sans = (projets_sans_aubaine / stats['total_projets'] * 100) if stats['total_projets'] > 0 else 0
            st.metric("Sans Aubaine", projets_sans_aubaine, f"{pct_sans:.1f}%")
        with col5:
            # Ratio des aubaines multiples parmi celles avec aubaines
            if stats['projets_avec_aubaine'] > 0:
                ratio_mult = (stats['aubaines_multiples'] / stats['projets_avec_aubaine'] * 100)
                st.metric("% Multiples/Avec Aubaines", f"{ratio_mult:.1f}%")
            else:
                st.metric("% Multiples/Avec Aubaines", "0%")
        
        # Détail par type d'aubaine
        st.subheader("DÉTAIL - Par Type d'Aubaine")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            pct_sim = (stats['aubaine_similarite'] / stats['total_projets'] * 100) if stats['total_projets'] > 0 else 0
            st.metric("Similarité", stats['aubaine_similarite'], f"{pct_sim:.1f}%")
        with col2:
            pct_temp = (stats['aubaine_temporelle'] / stats['total_projets'] * 100) if stats['total_projets'] > 0 else 0
            st.metric("Temporelle", stats['aubaine_temporelle'], f"{pct_temp:.1f}%")
        with col3:
            pct_trl = (stats['aubaine_trl'] / stats['total_projets'] * 100) if stats['total_projets'] > 0 else 0
            st.metric("TRL", stats['aubaine_trl'], f"{pct_trl:.1f}%")
        with col4:
            pct_part = (stats['aubaine_partenaire'] / stats['total_projets'] * 100) if stats['total_projets'] > 0 else 0
            st.metric("Partenaire", stats['aubaine_partenaire'], f"{pct_part:.1f}%")
        
        # Aubaines multiples détaillées
        st.subheader("AUBAINES MULTIPLES - Répartition")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("2 Aubaines", stats['aubaines_doubles'])
        with col2:
            st.metric("3 Aubaines", stats['aubaines_triples'])
        with col3:
            st.metric("4 Aubaines", stats['aubaines_quadruples'])
        with col4:
            if stats['projets_avec_aubaine'] > 0:
                pct_multiples_parmi_aubaines = (stats['aubaines_multiples'] / stats['projets_avec_aubaine'] * 100)
                st.metric("% Multiples/Total Aubaines", f"{pct_multiples_parmi_aubaines:.1f}%")
            else:
                st.metric("% Multiples/Total Aubaines", "0%")
        
        # Filtres
        st.header("FILTRES - Sélection")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            types_aubaines = ['Similarité', 'Temporelle', 'TRL', 'Partenaire']
            types_selectionnes = st.multiselect(
                "Types d'aubaines à afficher",
                options=types_aubaines,
                default=types_aubaines
            )
        
        with col2:
            nb_aubaines_options = sorted(df_processed['nb_aubaines'].unique())
            nb_aubaines_selectionnes = st.multiselect(
                "Nombre d'aubaines par projet",
                options=nb_aubaines_options,
                default=nb_aubaines_options
            )
        
        with col3:
            annees_disponibles = sorted(df_processed['edition_anr'].dropna().unique())
            annees_selectionnees = st.multiselect(
                "Éditions ANR",
                options=annees_disponibles,
                default=annees_disponibles
            )
        
        # Options supplémentaires
        col1, col2 = st.columns(2)
        with col1:
            montrer_multiples_seulement = st.checkbox("Aubaines multiples uniquement", value=False)
        with col2:
            montrer_aubaines_seulement = st.checkbox("Projets avec aubaine uniquement", value=False)
        
        # Application des filtres
        df_filtered = df_processed.copy()
        
        # Filtre par type d'aubaine
        if len(types_selectionnes) < 4:
            condition_type = pd.Series([False] * len(df_filtered))
            if 'Similarité' in types_selectionnes:
                condition_type |= df_filtered['aubaine_similarite']
            if 'Temporelle' in types_selectionnes:
                condition_type |= df_filtered['aubaine_temporelle']
            if 'TRL' in types_selectionnes:
                condition_type |= df_filtered['aubaine_trl']
            if 'Partenaire' in types_selectionnes:
                condition_type |= df_filtered['aubaine_partenaire']
            df_filtered = df_filtered[condition_type]
        
        # Autres filtres
        df_filtered = df_filtered[
            (df_filtered['nb_aubaines'].isin(nb_aubaines_selectionnes)) &
            (df_filtered['edition_anr'].isin(annees_selectionnees))
        ]
        
        if montrer_multiples_seulement:
            df_filtered = df_filtered[df_filtered['nb_aubaines'] > 1]
        
        if montrer_aubaines_seulement:
            df_filtered = df_filtered[df_filtered['a_aubaine']]
        
        # Tableau des résultats
        st.header("PROJETS - Résultats")
        
        nb_total = len(df_processed)
        nb_filtre = len(df_filtered)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total analysé", nb_total)
        with col2:
            st.metric("Après filtres", nb_filtre)
        
        if len(df_filtered) > 0:
            colonnes_affichage = [
                'code_projet_anr', 'acronyme_anr', 'titre_anr',
                'cordis_id', 'acronyme_cordis', 'titre_cordis',
                'edition_anr', 'call_year', 'trl_anr', 'trl_cordis',
                'bert_score', 'tfidf_score', 'pct_siren_match',
                'nb_partners_anr', 'nb_partners_cordis',
                'aubaines_detectees', 'nb_aubaines',
                'aubaine_similarite', 'aubaine_temporelle', 'aubaine_trl', 'aubaine_partenaire'
            ]
            
            colonnes_existantes = [col for col in colonnes_affichage if col in df_filtered.columns]
            df_display = df_filtered[colonnes_existantes].copy()
            
            # Formatage
            if 'bert_score' in df_display.columns:
                df_display['bert_score'] = df_display['bert_score'].round(3)
            if 'tfidf_score' in df_display.columns:
                df_display['tfidf_score'] = df_display['tfidf_score'].round(3)
            if 'pct_siren_match' in df_display.columns:
                df_display['pct_siren_match'] = df_display['pct_siren_match'].round(1)
            
            st.dataframe(
                df_display,
                use_container_width=True,
                height=400,
                column_config={
                    "bert_score": st.column_config.NumberColumn("Score BERT", format="%.3f"),
                    "tfidf_score": st.column_config.NumberColumn("Score TF-IDF", format="%.3f"),
                    "pct_siren_match": st.column_config.NumberColumn("% Partenaires", format="%.1f"),
                    "nb_partners_anr": st.column_config.NumberColumn("Nb Part. ANR", format="%d"),
                    "nb_partners_cordis": st.column_config.NumberColumn("Nb Part. CORDIS", format="%d"),
                    "aubaines_detectees": st.column_config.TextColumn("Aubaines Détectées", width="large"),
                    "nb_aubaines": st.column_config.NumberColumn("Nb Aubaines", format="%d"),
                    "aubaine_similarite": st.column_config.CheckboxColumn("Similarité"),
                    "aubaine_temporelle": st.column_config.CheckboxColumn("Temporelle"),
                    "aubaine_trl": st.column_config.CheckboxColumn("TRL"),
                    "aubaine_partenaire": st.column_config.CheckboxColumn("Partenaire"),
                }
            )
            
            # Export
            st.subheader("EXPORT - Téléchargements")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Export données filtrées
                output_filtre = io.BytesIO()
                with pd.ExcelWriter(output_filtre, engine='openpyxl') as writer:
                    df_filtered[colonnes_existantes].to_excel(writer, sheet_name='Donnees_Filtrees', index=False)
                
                st.download_button(
                    label="Données Filtrées",
                    data=output_filtre.getvalue(),
                    file_name=f"aubaines_filtrees_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            with col2:
                # Export par type
                output_types = io.BytesIO()
                with pd.ExcelWriter(output_types, engine='openpyxl') as writer:
                    # Feuille principale
                    df_avec_aubaine = df_processed[df_processed['a_aubaine']]
                    if len(df_avec_aubaine) > 0:
                        df_avec_aubaine[colonnes_existantes].to_excel(writer, sheet_name='Toutes_Aubaines', index=False)
                    
                    # Par type
                    types_dict = {
                        'Similarite': 'aubaine_similarite',
                        'Temporelle': 'aubaine_temporelle',
                        'TRL': 'aubaine_trl',
                        'Partenaire': 'aubaine_partenaire'
                    }
                    
                    for nom, col in types_dict.items():
                        df_type = df_processed[df_processed[col]]
                        if len(df_type) > 0:
                            df_type[colonnes_existantes].to_excel(writer, sheet_name=f'Type_{nom}', index=False)
                    
                    # Aubaines multiples
                    df_multiples = df_processed[df_processed['nb_aubaines'] > 1]
                    if len(df_multiples) > 0:
                        df_multiples[colonnes_existantes].to_excel(writer, sheet_name='Aubaines_Multiples', index=False)
                
                st.download_button(
                    label="Par Types",
                    data=output_types.getvalue(),
                    file_name=f"aubaines_par_types_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            with col3:
                # Export complet
                output_complet = io.BytesIO()
                with pd.ExcelWriter(output_complet, engine='openpyxl') as writer:
                    df_processed[colonnes_existantes].to_excel(writer, sheet_name='Donnees_Completes', index=False)
                    
                    # Statistiques
                    stats_df = pd.DataFrame([
                        ['Total projets', stats['total_projets']],
                        ['Projets avec aubaine', stats['projets_avec_aubaine']],
                        ['Aubaines multiples', stats['aubaines_multiples']],
                        ['Aubaines similarité', stats['aubaine_similarite']],
                        ['Aubaines temporelles', stats['aubaine_temporelle']],
                        ['Aubaines TRL', stats['aubaine_trl']],
                        ['Aubaines partenaires', stats['aubaine_partenaire']]
                    ], columns=['Métrique', 'Valeur'])
                    stats_df.to_excel(writer, sheet_name='Statistiques', index=False)
                    
                    # Paramètres
                    params_df = pd.DataFrame([
                        ['Seuil score élevé', seuil_score_eleve],
                        ['Seuil partenaires (%)', seuil_partenaires],
                        ['Min partenaires', min_partenaires],
                        ['Date génération', datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
                    ], columns=['Paramètre', 'Valeur'])
                    params_df.to_excel(writer, sheet_name='Parametres', index=False)
                
                st.download_button(
                    label="Export Complet",
                    data=output_complet.getvalue(),
                    file_name=f"aubaines_complet_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            # Visualisations
            st.header("VISUALISATIONS")
            fig1, fig2, fig3, fig4 = create_visualizations_flat(df_filtered)
            
            if fig1 is not None:
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig1, use_container_width=True)
                with col2:
                    st.plotly_chart(fig2, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig3, use_container_width=True)
                with col2:
                    st.plotly_chart(fig4, use_container_width=True)
        
        else:
            st.warning("Aucun projet ne correspond aux filtres sélectionnés.")
    
    else:
        st.error("Impossible de traiter les données avec les paramètres actuels.")

def main():
    # En-tête principal
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("Détecteur d'Effets d'Aubaine ANR-CORDIS")
    st.markdown("**Application de détection automatique des effets d'aubaine entre projets ANR et CORDIS**")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sidebar pour les paramètres
    with st.sidebar:
        st.header("Paramètres de Détection")
        
        # Seuil similarité
        st.subheader("SIMILARITÉ")
        seuil_score_eleve = st.slider(
            "Seuil pour aubaine de similarité",
            min_value=0.3,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Projets avec score BERT OU TF-IDF > ce seuil"
        )
        
        # Paramètres partenaires
        st.subheader("PARTENAIRES")
        min_partenaires = st.selectbox(
            "Nombre minimum de partenaires",
            options=[1, 2, 3, 4, 5],
            index=1,
            help="Minimum de partenaires requis"
        )
        
        seuil_partenaires = st.slider(
            "% minimal de partenaires communs",
            min_value=0,
            max_value=100,
            value=60,
            step=5,
            help="Seuil pour détecter les aubaines de partenaires"
        )
        
        st.markdown("---")
        st.markdown("**Types d'Aubaines Détectés :**")
        st.markdown("• **SIMILARITÉ** : score > seuil")
        st.markdown("• **TEMPORELLE** : |édition_anr - call_year| ≤ 1")
        st.markdown("• **TRL** : trl_anr = trl_cordis")a
        st.markdown(f"• **PARTENAIRE** : % > {seuil_partenaires}% ET ≥{min_partenaires} partners")
        
        
    
    # Chargement des données
    st.header("CHARGEMENT DES DONNÉES")
    
    with st.spinner("Chargement en cours..."):
        df_original = load_data_with_upload()
    
    if df_original is not None:
        afficher_detection_aubaine_flat(df_original, seuil_score_eleve, seuil_partenaires, min_partenaires)
    else:
        st.stop()
    
    # Documentation
    with st.expander("Documentation et Aide"):
        st.markdown("""
        ## Guide d'Utilisation
        
        ### Objectif de l'Application
        Cette application détecte les **effets d'aubaine** entre projets financés par l'ANR (France) et CORDIS (Europe). 
        Un effet d'aubaine survient quand les mêmes équipes obtiennent des financements multiples pour des projets similaires.
        
        ### Types d'Aubaines Détectés
        
        **Similarité** : Projets avec résumés très similaires (scores BERT/TF-IDF élevés)
        - Indique potentiellement le même concept soumis aux deux programmes
        
        **Temporelle** : Projets financés à moins d'un an d'intervalle
        - Suggère une stratégie de soumission simultanée
        
        **TRL** : Projets au même niveau de maturité technologique
        - Indique possiblement le même développement technologique
        
        **Partenaire** : Projets partageant de nombreux partenaires communs
        - Révèle des collaborations récurrentes entre les mêmes équipes
        
        ### Utilisation des Paramètres
        
        **Seuil de similarité** : Plus le seuil est bas, plus vous détecterez d'aubaines de similarité
        **Paramètres partenaires** : Ajustez selon votre définition de "collaboration significative"
        
        ### Interprétation des Résultats
        
        **Aubaines multiples** : Projets cumulant plusieurs types d'aubaines sont particulièrement suspects
        **Filtres** : Utilisez-les pour vous concentrer sur des aspects spécifiques
        **Exports** : Organisent les résultats pour analyse approfondie
        
        ### Points d'Attention
        
        - Les aubaines ne sont pas forcément problématiques (collaborations légitimes)
        - L'analyse manuelle reste nécessaire pour confirmer les cas suspects
        - Les seuils peuvent être ajustés selon le contexte d'analyse
        """)

if __name__ == "__main__":
    main()
