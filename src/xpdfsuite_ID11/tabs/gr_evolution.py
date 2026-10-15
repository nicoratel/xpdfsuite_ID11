"""Onglet « G(r) Time evolution » : fit des pics de G(r) et évolution temporelle."""

import json
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from xpdfsuite_ID11.app_utilities import open_fig_in_browser
from xpdfsuite_ID11.peakfit import fit_peak_pseudovoigt


def render() -> None:
    st.markdown("##📊 PDF peaks Time evolution")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Select the PDF peaks to monitor")
        peak_positions = st.text_input("Sélection des ppositions de pics à suivre (format: X, Y, Z...)",
                value="2.3, 2.9",
                help="Entrez les positions de pics à suivre, séparées par une virgule.")
        
        # Conversion de la chaîne "1, 2" en tuple (1, 2)
        try:
            positions = tuple(float(x.strip()) for x in peak_positions.split(","))
        except ValueError:
            st.error("Veuillez entrer deux nombres valides séparés par une virgule.")

    with col2:
        st.markdown("### Select the fitting window for each peak")
        fit_window = st.number_input("Largeur de la fenêtre de fit autour de chaque pic",
                value=0.4,
                step=0.01,
                help="Entrez la largeur de la fenêtre de fit autour de chaque pic.")

    
    ######################"  Execution button to fit peaks and plot evolution  ########################"
    
    if st.button("Fit PDF peaks and plot evolution", type="primary"):
        res = st.session_state.get("res")
        
        if res is None:
            st.error("Please compute the PDF first in the previous tab.")
        else:
            peak_evolution = {}
            nb_frame_groups = len(res)
            progress_bar_peaks = st.progress(0, text="Fit des pics en cours...")
            for group_idx, (frame_group, data) in enumerate(res.items()):
                    r = res[frame_group]['r']
                    G = res[frame_group]['G']
                    time = res[frame_group]['time']
                    # reaction_time has the same length as the list of frame_groups, so we can use it to label the x-axis in the plot
                    time_value = time
                    peak_evolution[time_value] = {}
                    for peak_pos in positions:
                        print(f"Fitting peak at {peak_pos} Å for frame group {frame_group} (time={time_value})")
                        try:
                            popt, pcov, area = fit_peak_pseudovoigt(r, G, position=peak_pos, window=fit_window, plot=False)
                            peak_evolution[time_value][peak_pos] = {
                                'amplitude': popt[0],
                                'center': popt[1],
                                'fwhm': popt[2],
                                'eta': popt[3],
                                'bkg': popt[4],
                                'area': area
                            }
                        except Exception as e:
                            print(f"Peak fitting failed for frame group {frame_group} at peak position {peak_pos}: {e}")
                            peak_evolution[time_value][peak_pos] = None
                    progress_bar_peaks.progress((group_idx + 1) / nb_frame_groups, text=f"Fit des pics {group_idx + 1}/{nb_frame_groups}")

            progress_bar_peaks.empty()
            # make plotly plots for each peak position
            
            # Configuration des 4 sous-graphiques : (clé, titre de l'axe Y, ligne, colonne)
            panels = [
                ('amplitude', 'Peak Amplitude', 1, 1),
                ('area',      'Peak Area',      1, 2),
                ('fwhm',      'Peak FWHM',      2, 1),
                ('center',    'Peak Center',    2, 2),
            ]

            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[p[1] for p in panels],
                horizontal_spacing=0.1,
                vertical_spacing=0.15,
            )

            colors = px.colors.qualitative.Plotly  # une couleur par pic, cohérente entre les 4 graphiques

            for i, peak_pos in enumerate(positions):
                # Collecte des données pour ce pic
                times = []
                data = {'amplitude': [], 'area': [], 'fwhm': [], 'center': []}

                for time_value in sorted(peak_evolution.keys()):
                    peak_data = peak_evolution[time_value][peak_pos]
                    if peak_data is not None:
                        times.append(time_value)
                        for key in data:
                            data[key].append(peak_data[key])

                # Ajout d'une trace par sous-graphique
                for key, ylabel, row, col in panels:
                    fig.add_trace(
                        go.Scatter(
                            x=times,
                            y=data[key],
                            mode='lines+markers',
                            name=f'Peak at {peak_pos} Å',
                            legendgroup=f'peak_{peak_pos}',      # cliquer sur la légende masque le pic partout
                            showlegend=(row == 1 and col == 1),  # une seule entrée de légende par pic
                            line=dict(color=colors[i % len(colors)]),
                            marker=dict(symbol='circle'),
                        ),
                        row=row, col=col,
                    )

            # Axes : labels et grilles
            for key, ylabel, row, col in panels:
                fig.update_xaxes(title_text='Reaction Time (s)', showgrid=True, row=row, col=col)
                fig.update_yaxes(title_text=ylabel, showgrid=True, row=row, col=col)

            fig.update_layout(
                title_text='Evolution of PDF Peaks Over Time',
                height=800,
                width=1200,
                template='plotly_white',
                hovermode='x unified',
            )

            st.session_state["fig_pdf_peaks"] = fig
            st.session_state["pdf_peak_evolution"] = peak_evolution  # <-- à ajouter

    fig_pdf_peaks_display = st.session_state.get("fig_pdf_peaks")
    if fig_pdf_peaks_display is not None:
        st.plotly_chart(fig_pdf_peaks_display, use_container_width=True)
        if st.button("🪟 Ouvrir dans une fenêtre séparée", key="open_window_pdf_peaks"):
            
                
            html = fig_pdf_peaks_display.to_html(
                full_html=True,
                include_plotlyjs=True, 
                config={"responsive": True},
            )
            # Injection du titre de l'onglet
            fig_title = f"Fit pics G(r):{st.session_state.get('samplename', 'Unknown')} {st.session_state.get('iq_raw_data')['option']} "
            open_fig_in_browser(fig_pdf_peaks_display, tab_title=fig_title, filename=f"fit_peaks_gr_{st.session_state.get('samplename', 'Unknown')}.html")
    st.download_button(
        label="Download PDF peak evolution data as JSON",
        data=json.dumps(st.session_state.get("pdf_peak_evolution"), indent=4,default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)),
        file_name=f"PDF_peak_evolution_data_{st.session_state.get('samplename')}.json",
        mime="application/json",
        key="download_pdf_peak_evolution_data"
    )
