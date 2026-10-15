"""Onglet « I(q) Time evolution » : fit des pics de I(q) et évolution temporelle."""

import json
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from xpdfsuite_ID11.app_utilities import open_fig_in_browser
from xpdfsuite_ID11.peakfit import fit_peak_pseudovoigt


def render() -> None:
    st.markdown("## 📊 I(q) Time evolution")
    iq_raw_data = st.session_state.get("iq_raw_data")


    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Select the I(q) peaks to monitor")
        
        peak_positions = st.text_input("Sélection des ppositions de pics à suivre (format: X, Y, Z...)",
                value="1.0, 2.0",
                help="Entrez les positions de pics à suivre, séparées par une virgule.")
        
        # Conversion de la chaîne "1, 2" en tuple (1, 2)
        try:
            positions = tuple(float(x.strip()) for x in peak_positions.split(","))
        except ValueError:
            st.error("Veuillez entrer deux nombres valides séparés par une virgule.")

    with col2:
        st.markdown("### Select the fitting window for each peak")
        fit_windowiq = st.number_input("Largeur de la fenêtre dq de fit autour de chaque pic",
                value=0.4,
                step=0.01,
                help="Entrez la largeur de la fenêtre de fit (deltaQ) autour de chaque pic.")

    
    ######################"  Execution button to fit peaks and plot evolution  ########################"
    
    if st.button("Fit I(q) peaks and plot evolution", type="primary"):

        iq_raw_data = st.session_state.get("iq_raw_data")

        if iq_raw_data is None:
            st.error("Please plot I(q) data first in the previous tab.")
        else:
            peak_evolution = {}

            bins = iq_raw_data["bins"]
            total = len(bins) * len(positions)
            done = 0

            progress_bar = st.progress(0.0, text="Starting peak fitting...")

            for bin_data in bins:
                qsample = bin_data["qsample"]
                Isample = bin_data["Isample"]
                time_value = bin_data["time"]
                # reaction_time has the same length as the list of frame_groups,
                # so we can use it to label the x-axis in the plot
                peak_evolution[time_value] = {}

                for peak_pos in positions:
                    try:
                        popt, pcov, area = fit_peak_pseudovoigt(
                            qsample, Isample, position=peak_pos,
                            window=fit_windowiq, plot=False
                        )
                        peak_evolution[time_value][peak_pos] = {
                            'amplitude': popt[0],
                            'center': popt[1],
                            'fwhm': popt[2],
                            'eta': popt[3],
                            'bkg': popt[4],
                            'area': area,
                        }
                    except Exception as e:
                        print(f"Peak fitting failed for time {time_value} at peak position {peak_pos}: {e}")
                        peak_evolution[time_value][peak_pos] = None

                    done += 1
                    progress_bar.progress(
                        done / total,
                        text=f"Fitting peaks... {done}/{total} (t = {time_value} s, peak {peak_pos} Å)"
                    )

            progress_bar.empty()  # retire la barre une fois terminé
            st.session_state["peak_evolution"] = peak_evolution

            # make plotly plots for each peak position
            
            # Configuration des 4 sous-graphiques : (clé, titre de l'axe Y, ligne, colonne)
            panels = [
                ('amplitude', 'Peak Amplitude', 1, 1),
                ('area',      'Peak Area',      1, 2),
                ('fwhm',      'Peak FWHM',      2, 1),
                ('center',    'Peak Center',    2, 2),
            ]

            fig2 = make_subplots(
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
                    fig2.add_trace(
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
                fig2.update_xaxes(title_text='Reaction Time (s)', showgrid=True, row=row, col=col)
                fig2.update_yaxes(title_text=ylabel, showgrid=True, row=row, col=col)

            fig2.update_layout(
                title_text='Evolution of PDF Peaks Over Time',
                height=800,
                width=1200,
                template='plotly_white',
                hovermode='x unified',
            )

            st.session_state["fig2_iq_peaks"] = fig2

    fig2_display = st.session_state.get("fig2_iq_peaks")
    if fig2_display is not None:
        st.plotly_chart(fig2_display, use_container_width=True)
        if st.button("🪟 Ouvrir dans une fenêtre séparée", key="open_window_iq_peaks"):
            html = fig2_display.to_html(
                full_html=True,
                include_plotlyjs=True, 
                config={"responsive": True},
            )
            # Injection du titre de l'onglet
            fig_title = f"Fit pics I(Q):{st.session_state.get('samplename', 'Unknown')} {st.session_state.get('iq_raw_data')['option']} "
            open_fig_in_browser(fig2_display, tab_title=fig_title, filename=f"fit_peaks_iq_{st.session_state.get('samplename', 'Unknown')}.html")

    st.download_button(
        label="Download I(q) peak evolution data as JSON",
        data=json.dumps(st.session_state.get("peak_evolution"), indent=4,default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)),
        file_name=f"IQ_peak_evolution_data_{st.session_state.get('samplename')}.json",
        mime="application/json",
        key="download_iq_peak_evolution_data"
    )
