import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

T = pd.read_csv("resultados/metricas_test.csv"); S = pd.read_csv("resultados/significancia.csv")
R = pd.read_csv("resultados/resumen_mejores.csv")
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=14, spaceBefore=10)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=11.5, spaceBefore=8)
B = ParagraphStyle("B", parent=ss["Normal"], fontSize=9.5, leading=13.5, alignment=4)
SM = ParagraphStyle("SM", parent=B, fontSize=8, leading=10.5, textColor=colors.HexColor("#444444"))
def P(t, s=B): return Paragraph(t, s)

def tbl(data, widths, hl_rows=()):
    t = Table(data, colWidths=widths, repeatRows=1)
    st = [("FONTSIZE", (0,0), (-1,-1), 8), ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1F3A5F")),
          ("TEXTCOLOR", (0,0), (-1,0), colors.white), ("ALIGN", (1,0), (-1,-1), "CENTER"),
          ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#999999")),
          ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F2F5F9")])]
    for r in hl_rows: st += [("BACKGROUND", (0,r), (-1,r), colors.HexColor("#D9EAD3")), ("FONTNAME", (0,r), (-1,r), "Helvetica-Bold")]
    t.setStyle(TableStyle(st)); return t

story = [P("Reporte Tecnico: Modelos de Aprendizaje Automatico para Prediccion Clinica", ss["Title"]),
         P("Tareas: Respuesta Patologica Completa (pCR), Supervivencia Global (OS) y Supervivencia Libre de Enfermedad (DFS)", SM),
         Spacer(1, 6)]

story += [P("1. Datos y metodologia", H1),
 P("Se usaron tres archivos: <b>train-data-pCR</b> (463 pacientes, 15.6 % pCR positiva), <b>survival-data-OS</b> (564 pacientes con desenlace conocido, 23.8 % de eventos) y "
   "<b>survival-data-DFS</b> (556 pacientes con desenlace conocido, 32.9 % de eventos). Las variables son clinicas y patologicas (estado menopausico, edad, cT, cN, histologia, grado, ER, PR, Ki67, HER2). "
   "En OS y DFS tambien se incluyen pCR, ypT y ypN como predictores; en la tarea pCR no se incluyen, porque se conocen despues de la cirugia. "
   "Los desenlaces OS y DFS ya venian codificados como 0/1 y no incluyen tiempo de seguimiento, por lo que se trataron como clasificacion binaria (no se ajustaron modelos de supervivencia con censura). "
   "Se eliminaron las filas con desenlace faltante (8 en OS, 16 en DFS)."),
 P("<b>Particion.</b> 80 % entrenamiento / 20 % prueba, estratificada por clase (semilla 42): 93 pacientes de prueba en pCR (14 positivos), 113 en OS y 112 en DFS. "
   "Toda la seleccion y la validacion cruzada se hicieron solo sobre el 80 % de entrenamiento; el conjunto de prueba se uso una unica vez para reportar metricas."),
 P("<b>Preprocesamiento (dentro de cada fold, sin fuga de datos).</b> Imputacion (mediana / moda), estandarizacion de numericas y one-hot de categoricas."),
 P("<b>Modelos (5).</b> Regresion Logistica, Random Forest, SVM (kernel RBF), XGBoost (300 arboles, profundidad 3, tasa de aprendizaje 0.05) y Red Neuronal MLP (32-16). Los modelos con soporte usan <i>class_weight='balanced'</i>."),
 P("<b>Umbral de decision.</b> Con clases desbalanceadas, el umbral 0.5 hace que algunos modelos casi nunca predigan la clase positiva. Cada modelo elige su umbral maximizando F1 con predicciones "
   "out-of-fold dentro del entrenamiento. Accuracy, Precision, Recall y F1 dependen de ese umbral; AUC y Brier no."),
 P("<b>Validacion cruzada y significancia.</b> CV estratificada 5 folds x 5 repeticiones (25 remuestreos, mismos folds para todos los modelos) sobre el entrenamiento, con AUC por fold. "
   "Pruebas: Friedman global; t pareada corregida de Nadeau-Bengio (prueba principal, corrige la dependencia entre folds solapados); Wilcoxon pareado como contraste no parametrico; "
   "correccion de Holm por comparaciones multiples (4 comparaciones por tarea). Un modelo se considera significativamente mejor solo si su AUC medio es mayor <b>y</b> el p-valor de Holm < 0.05."),
 P("<b>Criterio de seleccion del mejor modelo.</b> Mayor AUC medio en validacion cruzada (prioridad clinica a la discriminacion), revisando F1 y Brier. "
   "No se eligio con el conjunto de prueba para no sesgar la estimacion final.")]

names = {"pCR": "pCR (Respuesta Patologica Completa)", "OS": "OS (Supervivencia Global)", "DFS": "DFS (Supervivencia Libre de Enfermedad)"}
story += [PageBreak(), P("2. Resultados en el conjunto de prueba", H1),
          P("Fila resaltada = mejor modelo segun AUC medio en validacion cruzada. 'AUC CV' es media +/- desviacion estandar de 25 folds.", SM)]
for tk in ["pCR", "OS", "DFS"]:
    d = T[T.Tarea == tk].reset_index(drop=True); best = R[R.Tarea == tk]["Mejor modelo"].iloc[0]
    rows = [["Modelo", "Accuracy", "Precision", "Recall", "F1", "AUC", "Brier", "Umbral", "AUC CV"]]
    for _, r in d.iterrows():
        rows.append([r.Modelo, f"{r.Accuracy:.3f}", f"{r.Precision:.3f}", f"{r.Recall:.3f}", f"{r.F1:.3f}",
                     f"{r.AUC:.3f}", f"{r.Brier:.3f}", f"{r.Umbral:.2f}", f"{r.CV_AUC_media:.3f}+/-{r.CV_AUC_sd:.3f}"])
    hl = [i+1 for i, m in enumerate(d.Modelo) if m == best]
    story += [P(names[tk], H2), tbl(rows, [3*cm, 1.7*cm, 1.7*cm, 1.6*cm, 1.4*cm, 1.4*cm, 1.5*cm, 1.5*cm, 3*cm], hl), Spacer(1, 6)]

story += [P("3. Significancia estadistica (mejor modelo vs. los demas, AUC en CV)", H1),
          P("Dif. = AUC medio del mejor menos el del comparado. p(N-B) = t corregida de Nadeau-Bengio sin ajustar; p Holm = ajustado por comparaciones multiples. "
            "Los p-valores de Wilcoxon sobre 25 folds solapados son optimistas (anticonservadores) y se muestran solo como apoyo.", SM)]
for tk in ["pCR", "OS", "DFS"]:
    d = S[S.Tarea == tk]; fr = R[R.Tarea == tk]["Friedman p"].iloc[0]
    rows = [["Mejor", "vs.", "Dif. AUC", "p (N-B)", "p Holm (N-B)", "p Holm (Wilcoxon)", "Signif."]]
    for _, r in d.iterrows():
        rows.append([r.Mejor, r["Comparado con"], f"{r['Dif. media AUC']:+.3f}", f"{r['p (N-B)']:.4f}",
                     f"{r['p_Holm (N-B)']:.4f}", f"{r['p_Holm (Wilcoxon)']:.4f}", "Si" if r["Significativo (Holm N-B<0.05)"] else "No"])
    story += [P(f"{names[tk]} - Friedman p = {fr:.2e}", H2), tbl(rows, [2.4*cm, 2.6*cm, 1.8*cm, 1.9*cm, 2.4*cm, 3*cm, 1.5*cm]), Spacer(1, 4)]

story += [PageBreak(), P("4. Conclusiones", H1),
 P("<b>pCR - Mejor modelo: Regresion Logistica</b> (AUC CV 0.765; AUC prueba 0.745; F1 prueba 0.392). Supera significativamente a Random Forest (p Holm = 0.0034), "
   "XGBoost (0.0013) y MLP (0.027), pero <b>no</b> a SVM (p Holm = 0.097; diferencia de 0.042 en AUC). Veredicto: superioridad significativa sobre 3 de 4 competidores, no sobre todos. "
   "Nota: en el conjunto de prueba Random Forest obtuvo mejor AUC (0.778) y F1 (0.500), pero esa muestra tiene solo 14 casos positivos y en CV Random Forest rinde 0.078 AUC menos, de modo que se prioriza la evidencia de CV."),
 P("<b>OS - Mejor modelo: SVM</b> (AUC CV 0.642; AUC prueba 0.707; F1 prueba 0.481). Solo supera significativamente a la red neuronal MLP (p Holm = 0.006). "
   "Frente a Regresion Logistica, Random Forest y XGBoost las diferencias (0.014 a 0.019 de AUC) <b>no son estadisticamente significativas</b> (p Holm = 1.0). "
   "Los cuatro modelos distintos del MLP son estadisticamente equivalentes; el AUC absoluto es modesto (0.62 a 0.64 en CV)."),
 P("<b>DFS - Mejor modelo: Random Forest</b> (AUC CV 0.679; F1 CV 0.524; AUC prueba 0.665; F1 prueba 0.554, el mas alto). Solo supera significativamente al MLP (p Holm = 0.0004); "
   "contra Regresion Logistica, SVM y XGBoost no hay diferencia significativa (p Holm = 1.0)."),
 P("<b>Veredicto global.</b> En ninguna de las tres tareas el modelo elegido supera con p < 0.05 a <i>todos</i> los demas. Se puede afirmar que es mejor que los modelos senalados arriba; "
   "frente al resto la evidencia indica empate estadistico, por lo que la eleccion se apoya ademas en parsimonia (la Regresion Logistica es la mas interpretable). "
   "La red neuronal MLP fue el peor modelo en OS y DFS (AUC en CV cercano a 0.5) y en pCR quedo entre los dos peores."),
 P("5. Limitaciones", H1),
 P("(i) Muestras pequenas: conjuntos de prueba de 93 a 113 pacientes, con pocos eventos en pCR, por lo que las metricas de prueba tienen amplia incertidumbre y no deben usarse para elegir modelo. "
   "(ii) OS y DFS se modelaron como clasificacion binaria sin tiempo de seguimiento ni censura; no se uso analisis de supervivencia (Cox, RSF). "
   "(iii) Sin ajuste exhaustivo de hiperparametros (configuraciones fijas razonables). "
   "(iv) La MLP con tan pocos datos tiende a sobreajustar. "
   "(v) Sin validacion externa. (vi) En OS/DFS, ypT y ypN solo estan disponibles tras la cirugia; los modelos son pronosticos post-operatorios."),
 P("Reproducibilidad: <i>pipeline_clinico.py</i> (semilla 42) genera los CSV en <i>resultados/</i>; <i>make_report.py</i> construye este PDF.", SM)]

SimpleDocTemplate("reporte_tecnico.pdf", pagesize=letter, leftMargin=2*cm, rightMargin=2*cm, topMargin=1.8*cm, bottomMargin=1.8*cm,
                  title="Reporte Tecnico ML Clinico").build(story)
