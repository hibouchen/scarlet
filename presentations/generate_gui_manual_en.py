#!/usr/bin/env python3
"""Create the English SCARLET GUI manual from the validated French layout."""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "presentations" / "SCARLET_manuel_interface_graphique_fr.pptx"
OUTPUT = ROOT / "presentations" / "SCARLET_GUI_user_manual_en.pptx"


TRANSLATIONS = {
    "Manuel de l’interface graphique": "Graphical User Interface Manual",
    "Du dossier brut aux courbes I(q) — avec intégration et stitching sectoriels": "From raw data to I(q) curves — including sector integration and stitching",
    "Ce guide accompagne l’opérateur page par page :": "This guide takes the operator through the interface page by page:",
    "•  préparer et valider le workflow": "•  prepare and validate the workflow",
    "•  réduire en isotrope ou par secteurs": "•  reduce isotropic or sector data",
    "•  raccorder puis comparer les courbes": "•  stitch and compare the curves",
    "Version 0.4.0 · 17 novembre 2026": "Version 0.4.0 · November 17, 2026",
    "VUE D’ENSEMBLE": "OVERVIEW",
    "Le parcours en 6 étapes": "The 6-step workflow",
    "SCARLET · Manuel de l’interface graphique": "SCARLET · Graphical User Interface Manual",
    "Choisir les dossiers et créer ou ouvrir un workflow.": "Select the folders and create or open a workflow.",
    "Vérifier les fichiers, métadonnées et masques.": "Review the files, metadata, and masks.",
    "Préparer centres, ROI, transmissions et flatfields.": "Prepare beam centers, ROI, transmissions, and flatfields.",
    "Réduire en isotrope et, si besoin, par secteurs.": "Run isotropic and, when needed, sector reduction.",
    "Raccorder les configurations pour chaque famille de courbes.": "Stitch the configurations for each curve family.",
    "Superposer, comparer et contrôler les courbes I(q).": "Overlay, compare, and review the I(q) curves.",
    "CONSEIL": "TIP",
    "L’ordre des pages suit le travail réel. La couleur des statuts indique où intervenir avant de lancer un calcul.": "The page order follows the actual workflow. Status colors show where action is required before running a calculation.",
    "Les repères permanents": "Persistent interface landmarks",
    "Les six pages restent accessibles à gauche.": "The six workflow pages remain available on the left.",
    "Nom, instrument, nombre d’acquisitions et bouton Save.": "Name, instrument, acquisition count, and Save button.",
    "Zone de travail": "Workspace",
    "Tableaux, visualisations et actions de la page.": "Tables, visualizations, and page actions.",
    "Trace des opérations et messages utiles au diagnostic.": "Operation log and messages useful for troubleshooting.",
    "Créer ou ouvrir un workflow": "Create or open a workflow",
    "À renseigner": "Required information",
    "•  Raw data : dossier des acquisitions": "•  Raw data: acquisition folder",
    "•  Output : dossier des produits": "•  Output: product folder",
    "•  Instrument et stratégie de transmission": "•  Instrument and transmission strategy",
    "•  Experiment ID : nom court et stable": "•  Experiment ID: short, stable name",
    "Scan acquisitions crée le workflow. Open workflow… reprend un fichier sauvegardé.": "Scan acquisitions creates the workflow. Open workflow… resumes a saved file.",
    "Contrôler les acquisitions": "Review the acquisitions",
    "Avant de poursuivre": "Before proceeding",
    "•  Corriger Sample, Mode, Entity ou Thickness par double-clic.": "•  Double-click to correct Sample, Mode, Entity, or Thickness.",
    "•  Utiliser les filtres pour isoler une configuration ou un statut.": "•  Use the filters to isolate a configuration or status.",
    "•  Exclure uniquement les fichiers qui ne doivent pas entrer dans le traitement.": "•  Exclude only files that must not enter processing.",
    "•  Vérifier la ligne verte Ready for processing.": "•  Check for the green Ready for processing line.",
    "Un clic sur une ligne charge son détecteur à droite.": "Click a row to load its detector view on the right.",
    "Visualiser et enregistrer un masque": "View and save a mask",
    "Méthode": "Method",
    "•  Sélectionner une acquisition représentative.": "•  Select a representative acquisition.",
    "•  Choisir le détecteur dans la vue.": "•  Select the detector in the viewer.",
    "•  Dessiner ou ajuster les zones masquées.": "•  Draw or adjust the masked regions.",
    "•  Cliquer Save and attach mask…": "•  Click Save and attach mask…",
    "À RETENIR": "REMEMBER",
    "Convention SCARLET : 1 = masqué, 0 = valide. Le masque est associé à la configuration.": "SCARLET convention: 1 = masked, 0 = valid. The mask is assigned to the configuration.",
    "Préparer les configurations": "Prepare the configurations",
    "Deux niveaux de contrôle": "Two levels of review",
    "Tableau principal": "Main table",
    "Centres, ROI, masques, flatfield et statut global par configuration.": "Beam centers, ROI, masks, flatfield, and overall status for each configuration.",
    "Valeurs effectives de l’eau et de la cellule vide, placées sous le tableau des configurations.": "Effective water and empty-cell values, shown below the configuration table.",
    "Une configuration est prête quand toutes les colonnes sont complètes.": "A configuration is ready when every column is complete.",
    "Centre du faisceau et ROI": "Beam center and ROI",
    "•  Charger l’empty beam.": "•  Load the empty beam.",
    "•  Détecter automatiquement ou ajuster X/Y.": "•  Detect automatically or adjust X/Y.",
    "•  Sauvegarder pour chaque détecteur.": "•  Save the center for each detector.",
    "•  Passer Edit à Transmission ROI.": "•  Set Edit to Transmission ROI.",
    "•  Détecter ou saisir x0, x1, y0, y1.": "•  Detect or enter x0, x1, y0, y1.",
    "•  Vérifier la zone sur l’image avant Save.": "•  Check the region on the image before Save.",
    "Masques, flatfields et transmissions": "Masks, flatfields, and transmissions",
    "Onglets de préparation": "Preparation tabs",
    "•  Transmission : réutiliser une configuration source si nécessaire.": "•  Transmission: reuse a source configuration when needed.",
    "•  Mask : attacher un bundle ou rechercher un masque compatible.": "•  Mask: attach a bundle or find a compatible mask.",
    "•  Flatfield : créer, choisir un fichier ou réutiliser une source.": "•  Flatfield: create one, select a file, or reuse a source.",
    "ATTENTION": "CAUTION",
    "Une modification de masque ou de transmission peut rendre un flatfield obsolète : reconstruisez-le si le statut devient stale.": "A mask or transmission change can make a flatfield stale. Rebuild it when its status becomes stale.",
    "Comprendre les statuts": "Understand the statuses",
    "Tous les prérequis sont présents.": "All prerequisites are available.",
    "Au moins un élément manque.": "At least one item is missing.",
    "Un produit doit être reconstruit.": "A product must be rebuilt.",
    "Checklist avant Processing": "Checklist before Processing",
    "•  un centre de faisceau par détecteur": "•  one beam center per detector",
    "•  une ROI de transmission valide": "•  a valid transmission ROI",
    "•  des masques assignés": "•  assigned masks",
    "•  un flatfield présent et non stale": "•  an available, non-stale flatfield",
    "•  une transmission pour chaque échantillon": "•  a transmission for every sample",
    "•  une épaisseur renseignée si disponible": "•  a thickness value when available",
    "•  des fichiers de référence cohérents": "•  consistent reference files",
    "•  aucun message bloquant dans Acquisitions": "•  no blocking message in Acquisitions",
    "Réduire les acquisitions": "Reduce the acquisitions",
    "Traitement standard": "Standard processing",
    "•  Sélectionner une ou plusieurs acquisitions.": "•  Select one or more acquisitions.",
    "•  Régler le nombre de bins par détecteur.": "•  Set the number of bins for each detector.",
    "•  Choisir l’échelle q : Linear ou Logarithmic.": "•  Choose the q scale: Linear or Logarithmic.",
    "•  Lancer Process selected ou Process pending.": "•  Run Process selected or Process pending.",
    "La réduction s’exécute en arrière-plan. Les statuts passent de pending à processed.": "Reduction runs in the background. Statuses change from pending to processed.",
    "Intégration sectorielle": "Sector integration",
    "Réglages": "Settings",
    "•  Activer Enable sector integration.": "•  Enable sector integration.",
    "•  Preset : parallèle/perpendiculaire ou direction unique.": "•  Preset: parallel/perpendicular or a single direction.",
    "•  Main direction : angle sur le détecteur.": "•  Main direction: angle on the detector.",
    "•  Sector width : largeur angulaire totale.": "•  Sector width: full angular width.",
    "•  Combine opposite : moyenne la direction à 180°.": "•  Combine opposite: average the direction 180° away.",
    "LECTURE DE LA VUE": "READING THE VIEW",
    "Rouge : direction principale. Bleu : perpendiculaire. La vue suit la ligne sélectionnée.": "Red: main direction. Blue: perpendicular. The preview follows the selected table row.",
    "Raccorder l’isotrope ou les secteurs": "Stitch isotropic or sector curves",
    "•  Isotropic : courbe moyenne classique.": "•  Isotropic: standard averaged curve.",
    "•  All common sectors : tous les secteurs compatibles.": "•  All common sectors: every compatible sector.",
    "•  Sector: … : un secteur nommé uniquement.": "•  Sector: …: one named sector only.",
    "Conditions": "Requirements",
    "Toutes les configurations doivent être processed et partager une définition de secteur identique.": "Every configuration must be processed and share an identical sector definition.",
    "NOM DU FICHIER": "FILE NAME",
    "Examiner et comparer les résultats": "Review and compare the results",
    "Utilisation": "Usage",
    "•  Sélection multiple pour superposer les courbes.": "•  Select multiple rows to overlay curves.",
    "•  Le type distingue reduced, sector, stitched et sector stitched.": "•  Type distinguishes reduced, sector, stitched, and sector stitched.",
    "•  Utiliser Search pour filtrer les fichiers.": "•  Use Search to filter the files.",
    "•  La légende reprend le nom de chaque sortie.": "•  The legend uses each output file name.",
    "Comparer parallèle et perpendiculaire révèle l’anisotropie.": "Comparing parallel and perpendicular curves reveals anisotropy.",
    "TRAÇABILITÉ": "TRACEABILITY",
    "Sauvegarde et fichiers produits": "Saving and output files",
    "Sauvegarder le workflow": "Save the workflow",
    "•  Bouton Save dans la barre latérale": "•  Save button in the sidebar",
    "•  Ctrl+S pour la sauvegarde courante": "•  Ctrl+S to save the current workflow",
    "•  File → Save workflow as… pour créer une copie": "•  File → Save workflow as… to create a copy",
    "•  Le workflow conserve acquisitions, réglages et liens vers les produits": "•  The workflow stores acquisitions, settings, and links to products",
    "Sauvegarder après chaque étape de préparation importante.": "Save after every major preparation step.",
    "Les sorties texte contiennent Q, I, dI et dQ.": "Text outputs contain Q, I, dI, and dQ.",
    "AIDE": "HELP",
    "Diagnostiquer sans perdre le fil": "Troubleshoot without losing track",
    "Vert": "Green",
    "Prêt / traité": "Ready / processed",
    "Poursuivre vers l’étape suivante.": "Proceed to the next step.",
    "Orange": "Orange",
    "Avertissement / stale": "Warning / stale",
    "Lire le détail et confirmer la cohérence.": "Read the details and confirm consistency.",
    "Rouge": "Red",
    "Blocage / missing": "Blocked / missing",
    "Compléter l’entrée indiquée avant de relancer.": "Complete the indicated input before trying again.",
    "Où chercher ?": "Where should I look?",
    "•  Le texte sous le résumé de page": "•  The text below the page summary",
    "•  Les infobulles des cellules de statut": "•  Tooltips on status cells",
    "•  Le panneau Activity en bas de la fenêtre": "•  The Activity panel at the bottom of the window",
    "•  Le fichier absent ou incohérent mentionné dans le message": "•  The missing or inconsistent file named in the message",
    "Corriger la première cause signalée, puis rafraîchir la page concernée.": "Fix the first reported cause, then refresh the relevant page.",
    "RÉSUMÉ": "SUMMARY",
    "Checklist opérateur": "Operator checklist",
    "Dossiers, instrument et identifiant vérifiés": "Folders, instrument, and identifier checked",
    "Fichiers classés, épaisseurs et masques contrôlés": "Files classified; thicknesses and masks reviewed",
    "Centres, ROI, transmissions et flatfields ready": "Beam centers, ROI, transmissions, and flatfields ready",
    "Bins, échelle q et secteurs choisis": "Bins, q scale, and sectors selected",
    "Famille de courbes, référence et facteur validés": "Curve family, reference, and factor validated",
    "Courbes superposées, cohérentes et sauvegardées": "Curves overlaid, consistent, and saved",
    "BONNE PRATIQUE": "GOOD PRACTICE",
    "Raccourci mental : préparer → vérifier → traiter → raccorder → comparer → sauvegarder.": "Mental shortcut: prepare → review → process → stitch → compare → save.",
}


def _translate_paragraph(paragraph) -> bool:
    translated = TRANSLATIONS.get(paragraph.text)
    if translated is None:
        return False
    if paragraph.runs:
        paragraph.runs[0].text = translated
        for run in paragraph.runs[1:]:
            run.text = ""
    else:
        paragraph.text = translated
    return True


def main() -> int:
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Generate the French layout first: {SOURCE}")
    presentation = Presentation(SOURCE)
    translated_count = 0
    for slide in presentation.slides:
        for shape in slide.shapes:
            if not getattr(shape, "has_text_frame", False):
                continue
            for paragraph in shape.text_frame.paragraphs:
                translated_count += int(_translate_paragraph(paragraph))

    presentation.core_properties.title = "SCARLET — Graphical User Interface Manual"
    presentation.core_properties.subject = (
        "Operator guide for the SANS workflow, including sector integration and stitching"
    )
    presentation.core_properties.comments = (
        "English manual generated from the validated SCARLET interface layout."
    )
    presentation.save(OUTPUT)
    print(f"{OUTPUT} ({translated_count} text elements translated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
